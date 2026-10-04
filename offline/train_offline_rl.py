import argparse
import hydra
import torch
import numpy as np
import os
import gym
import d4rl 
import time 
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import OfflineDataset
from src.data.buffer import GPUReplayBuffer
from src.networks.base_sac_network import Actor, Critic
from src.algorithms.sac import SAC
from src.utils.logger import Logger
from src.uncertainty.uncertainty_rnd_rff import RNDGRFFUncertainty as RNDRFFUncertainty


def _parse_config_path(argv=None):
    parser = argparse.ArgumentParser(
        description="Train offline RL from a Hydra YAML config file."
    )
    parser.add_argument("config_path", nargs="?", help="Path to a YAML config file.")
    parser.add_argument(
        "--config",
        dest="config_option",
        help="Path to a YAML config file.",
    )
    args = parser.parse_args(argv)

    if args.config_path and args.config_option:
        parser.error("provide either positional config_path or --config, not both")

    config_path = args.config_path or args.config_option
    if config_path is None:
        return None

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if path.suffix != ".yaml":
        parser.error("config path must point to a .yaml file")
    if not path.is_file():
        parser.error(f"config file does not exist: {path}")

    return path


def _config_path_to_hydra_argv(config_path):
    return [
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
    ]

def evaluate_parallel(actor, normalizer, eval_envs, num_episodes=10, device="cpu"):

    obs = eval_envs.reset() 
    
    episode_returns = np.zeros(num_episodes)
    finished_mask = np.zeros(num_episodes, dtype=bool)
    
    while not all(finished_mask):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            obs_norm = normalizer.normalize(obs_tensor)
            _, _, action_mean = actor.sample(obs_norm)
            actions = action_mean.cpu().numpy()
            
        obs, rewards, dones, _ = eval_envs.step(actions)
        
        for i in range(num_episodes):
            if not finished_mask[i]:
                episode_returns[i] += rewards[i]
                if dones[i]:
                    finished_mask[i] = True

    
    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    
    return avg_return, std_return


@hydra.main(config_path="config", config_name="train_default_config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    dataset = OfflineDataset(cfg, device)
    normalizer = dataset.get_normalizer()
    obs, act, rew, next_obs, done = dataset.get_data()
    
    print("[Pre-processing] Normalizing observations...")
    obs_tensor = torch.tensor(obs, device=device)
    next_obs_tensor = torch.tensor(next_obs, device=device)
    obs_normalized = normalizer.normalize(obs_tensor)
    next_obs_normalized = normalizer.normalize(next_obs_tensor)
    
    buffer = GPUReplayBuffer(obs_normalized, act, rew, next_obs_normalized, done, device)
    dataset_size = buffer.size
    steps_per_epoch = int(dataset_size / cfg.batch_size)
    max_steps = cfg.max_epochs * steps_per_epoch
    
    print(f"Dataset Size: {dataset_size} | Batch Size: {cfg.batch_size}")
    print(f"Total Training: {cfg.max_epochs} Epochs -> {max_steps} Steps")

    obs_dim = obs.shape[1]
    action_dim = act.shape[1]
    
    actor = Actor(obs_dim, action_dim, cfg.algorithm).to(device)
    critic = Critic(obs_dim, action_dim, cfg.algorithm).to(device)
    
    if cfg.uncertainty.name == "rnd_rff":
        uncertainty_module = RNDRFFUncertainty(obs_dim, action_dim, cfg.uncertainty, device)

    else:
        raise ValueError(f"Unknown uncertainty method: {cfg.uncertainty.name}")



    agent = SAC(actor, critic, uncertainty_module, cfg.algorithm, device)
    
    logger = Logger(cfg)
    
    print("Start Training...")

    rnd_epochs = cfg.get('rnd_training_epochs', 0)
    rnd_steps = cfg.get('rnd_training_steps', 0)
    use_epochs = rnd_epochs > 0
    
    if (use_epochs or rnd_steps > 0) and hasattr(agent, 'train_rnd_only'):
        print(f"\n=== Phase 1: Pre-training Uncertainty Module ({cfg.uncertainty.name}) ===")
        
        if use_epochs:
            total_rnd_steps = rnd_epochs * steps_per_epoch
            print(f"Plan: {rnd_epochs} Epochs -> {total_rnd_steps} Steps")
        else:
            total_rnd_steps = rnd_steps
            print(f"Plan: {total_rnd_steps} Fixed Steps")

        rnd_start_time = time.time()
        for i in range(1, total_rnd_steps + 1):
            batch = buffer.sample(cfg.batch_size)
            rnd_loss = agent.train_rnd_only(batch)
            
            log_freq = steps_per_epoch if use_epochs else 5000
            if log_freq < 1: log_freq = 1
            
            if i % log_freq == 0:
                current_epoch = i // steps_per_epoch if use_epochs else "N/A"
                print(f"RND Pre-train | Step {i}/{total_rnd_steps} (Epoch {current_epoch}) | Loss: {rnd_loss:.6f}")
        
        print(f"RND Pre-training Finished. (Time: {time.time() - rnd_start_time:.1f}s)")
        
        for param in uncertainty_module.parameters():
            param.requires_grad = False
    

    print("\n=== Phase 2: Start SAC Training ===")
    
    best_norm_score = -float('inf')
    
    update_func = agent.update
    buffer_sample = buffer.sample
    batch_size = cfg.batch_size
    log_every_epochs = cfg.log_every_epochs
    eval_every_epochs = cfg.eval_every_epochs
    
    print("\n[Setup] Initializing Vectorized Evaluation Environments...")
    num_eval_episodes = 10
    
    def make_env_fn(rank):
        def _thunk():
            env = gym.make(cfg.env.dataset_name)
            return env
        return _thunk

    try:
        eval_envs = gym.vector.AsyncVectorEnv([make_env_fn(i) for i in range(num_eval_episodes)])
    except Exception as e:
        print(f"[Warning] Async failed, using Sync: {e}")
        eval_envs = gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(num_eval_episodes)])

    start_time = time.time()
    
    try: 
        for step in range(1, max_steps + 1):
            batch = buffer_sample(batch_size)
            
            current_epoch = step // steps_per_epoch
            is_epoch_end = (step % steps_per_epoch == 0)
            
            is_log_step = is_epoch_end and (current_epoch % log_every_epochs == 0)
            is_eval_step = is_epoch_end and (current_epoch % eval_every_epochs == 0)
            
            train_logs = update_func(batch, log_this_step=is_log_step)
            if is_log_step:
                current_time = time.time() - start_time
                train_logs['time'] = current_time
                train_logs['epoch'] = current_epoch
                logger.log_train(train_logs, step=step)

            if is_eval_step:
                avg_return, std_return = evaluate_parallel(
                    actor, 
                    normalizer, 
                    eval_envs, 
                    num_episodes=num_eval_episodes, 
                    device=device
                )
                
                dummy_env = gym.make(cfg.env.dataset_name)
                norm_score = dummy_env.get_normalized_score(avg_return) * 100
                dummy_env.close()
                
                eval_logs = {
                    'eval_return': avg_return,
                    'normalized_score': norm_score,
                    'epoch': current_epoch,
                    'time': time.time() - start_time
                }
                logger.log_eval(eval_logs, step=step)
                
                if norm_score > best_norm_score:
                    best_norm_score = norm_score
                    
                    save_dict = {
                        'step': step,
                        'best_norm_score': best_norm_score,
                        'actor': actor.state_dict(),
                        'critic': critic.state_dict(),
                        'normalizer': normalizer.state_dict(),
                        'config': cfg
                    }
                    
                    if agent.automatic_entropy_tuning:
                        save_dict['log_alpha'] = agent.log_alpha
                        save_dict['alpha_optimizer'] = agent.alpha_optimizer.state_dict()

                    if hasattr(uncertainty_module, 'state_dict'):
                        save_dict['uncertainty'] = uncertainty_module.state_dict()
                    
                    save_path = os.path.join(logger.checkpoint_dir, "best_model.pt")
                    torch.save(save_dict, save_path)
                    print(f"★ New Best Model Saved! ({norm_score:.2f})")

    finally:
        
        logger.close()

    save_path = os.path.join(logger.checkpoint_dir, "last_model.pt")
    torch.save(save_dict, save_path)
    print(f"★ Last Model Saved! ({norm_score:.2f})")
    if 'eval_envs' in locals():
        print("Closing evaluation environments...")
        eval_envs.close()


if __name__ == "__main__":
    config_path = _parse_config_path(sys.argv[1:])
    if config_path is not None:
        sys.argv = [sys.argv[0], *_config_path_to_hydra_argv(config_path)]
    main()
