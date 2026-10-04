import hydra
import torch
import numpy as np
import os
import warnings
import gym
import d4rl
from omegaconf import DictConfig, OmegaConf

from src.networks.base_sac_network import Actor
from src.utils.normalization import StateNormalizer

from src.uncertainty.uncertainty_rnd import RNDUncertainty
from src.uncertainty.uncertainty_rnd_rff import RNDRFFUncertainty
from src.uncertainty.base import BaseUncertainty

class DummyUncertainty(BaseUncertainty):
    def compute_uncertainty(self, o, a): return torch.zeros((o.shape[0], 1))
    def update(self, b): pass
    def state_dict(self): return {}
    def load_state_dict(self, s): pass

def load_uncertainty_module(cfg, obs_dim, action_dim, device):
    method = cfg.uncertainty.name
    
    
    if method == "rnd_rff":
        module = RNDRFFUncertainty(obs_dim, action_dim, cfg.uncertainty, device)
 
    return module

@hydra.main(config_path="config", config_name="eval_default_config", version_base=None)
def main(eval_cfg: DictConfig):
    device = torch.device(eval_cfg.device if torch.cuda.is_available() else "cpu")
    
    if eval_cfg.target_dir == "???":
        raise ValueError("Please provide 'target_dir'.")

    target_path = os.path.abspath(eval_cfg.target_dir)
    ckpt_path = os.path.join(target_path, "checkpoints", "last_model.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    train_cfg = checkpoint['config']
    
    env_name = train_cfg.env.dataset_name
    env = gym.make(env_name)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = Actor(obs_dim, action_dim, train_cfg.algorithm).to(device)
    state_dict = checkpoint['actor']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
        
    actor.load_state_dict(new_state_dict)
    actor.eval()
    
    normalizer = StateNormalizer(obs_dim, device=device)
    normalizer.load_state_dict(checkpoint['normalizer'])
    
    uncertainty_module = load_uncertainty_module(train_cfg, obs_dim, action_dim, device)
    if 'uncertainty' in checkpoint:
        uncertainty_module.load_state_dict(checkpoint['uncertainty'])
    

    eval_seeds = eval_cfg.get('eval_seeds', [42])
    if hasattr(eval_seeds, 'dtype'):
        eval_seeds = eval_seeds.tolist()
    else:
        eval_seeds = list(eval_seeds)

    n_episodes = eval_cfg.get('n_episodes', 10) 
    

    all_returns = []
    all_uncertainty_stats = []
    
    for seed in eval_seeds:
        try:
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except:
            pass 
            
        seed_returns = []
        
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            ep_ret = 0
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    obs_norm = normalizer.normalize(obs_tensor).unsqueeze(0)
                    
                    action_tensor = actor.evaluation(obs_norm)
                    action = action_tensor.cpu().numpy()[0]
                    
                    unc = uncertainty_module.compute_uncertainty(obs_norm, action_tensor)
                    all_uncertainty_stats.append(unc.item())

                next_obs, reward, done, _ = env.step(action)
                ep_ret += reward
                obs = next_obs
            
            norm_score = env.get_normalized_score(ep_ret) * 100
            seed_returns.append(norm_score)
            all_returns.append(norm_score)
        

    avg_score = np.mean(all_returns)
    std_score = np.std(all_returns)
    avg_unc = np.mean(all_uncertainty_stats) if all_uncertainty_stats else 0.0
    
    print("-" * 50)
    print(f"Environment: {env_name}")
    print(f"Seeds: {eval_seeds}")
    print(f"Total Episodes: {len(all_returns)}")
    print(f"Average Normalized Score: {avg_score:.2f} ± {std_score:.2f}")
    print("-" * 50)
    
    result_path = os.path.join(target_path, "final_eval_result.txt")
    with open(result_path, "w") as f:
        f.write(f"Environment: {env_name}\n")
        f.write(f"Algorithm: {train_cfg.algorithm.name} + {train_cfg.uncertainty.name}\n")
        f.write(f"Num Seeds: {len(eval_seeds)}\n")
        f.write(f"Average Score: {avg_score:.2f}\n")
        f.write(f"Std Score: {std_score:.2f}\n")
        f.write(f"Avg Uncertainty: {avg_unc:.4f}\n")
        

if __name__ == "__main__":
    main()