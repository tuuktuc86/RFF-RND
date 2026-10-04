import os
import shutil
from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
import math
from create_agent import create_agent

def get_max_global_updates(env_id: str) -> int:
    env_id_lower = env_id.lower()
    if "gravitar" in env_id_lower:
        return 8000

    if any(k in env_id_lower for k in ["venture", "frogger", "freeway"]):
        return 8000

    if any(k in env_id_lower for k in ["montezuma", "pitfall", "solaris"]):
        return 8000
    return 4500

def save_config_once(config_path, log_dir, filename="config_used.conf"):
    os.makedirs(log_dir, exist_ok=True)
    dst = os.path.join(log_dir, filename)
    if not os.path.exists(dst):
        shutil.copy(config_path, dst)
        print(f"[CONFIG] saved to {dst}")
    else:
        print(f"[CONFIG] already exists, skip: {dst}")

def main():
    default_config, config_path = load_config()
    for k, v in default_config.items():
        print(f"{k} = {v}")

    is_render=False
    agent, writer, log_dir, extra = create_agent(default_config)
    save_config_once(config_path, log_dir)
    
    env_cls = extra["env_cls"]
    env_id = extra["env_id"]
    max_global_updates= get_max_global_updates(env_id)
    output_size = extra["output_size"]
    num_worker = extra["num_worker"]
    num_step = extra["num_step"]
    sticky_action = extra["sticky_action"]
    action_prob = extra["action_prob"]
    life_done = extra["life_done"]
    pre_obs_norm_step = int(extra['pre_obs_norm_step'])
    gamma = extra["gamma"]
    int_gamma=extra["int_gamma"]
    reward_rms= extra['reward_rms']
    obs_rms = extra['obs_rms']
    discounted_reward = RewardForwardFilter(int_gamma)
    ext_coef = extra["ext_coef"]
    int_coef = extra["int_coef"]

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_cls(env_id, is_render, idx, child_conn,
                    sticky_action=sticky_action, p=action_prob, life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    for step in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            next_obs.append(s[3, :, :].reshape([1, 84, 84]))

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End to initalize...')

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1
        
        if global_update >= max_global_updates:
            print(f"[STOP] Reached max_global_steps: global_step={global_update} >= {max_global_updates}")
            break

        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(s[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)

            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))

            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)

        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)
        
        total_int_reward /= np.sqrt(reward_rms.var)
        
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

 
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)

        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        total_adv = int_adv * int_coef + ext_adv * ext_coef

        obs_rms.update(total_next_obs)

        agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy)

if __name__ == '__main__':
    main()
