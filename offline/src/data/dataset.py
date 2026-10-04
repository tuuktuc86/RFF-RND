import gym
import d4rl
import numpy as np
import torch
from src.utils.normalization import StateNormalizer

class OfflineDataset:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        env_name = cfg.env.dataset_name
        print(f"[Dataset] Loading D4RL dataset: {env_name}")
        
        env = gym.make(env_name)

        dataset = d4rl.qlearning_dataset(env)

        self.observations = dataset['observations'].astype(np.float32)
        self.actions = dataset['actions'].astype(np.float32)
        self.next_observations = dataset['next_observations'].astype(np.float32)
        
        self.rewards = dataset['rewards'].astype(np.float32).reshape(-1, 1)
        self.dones = dataset['terminals'].astype(np.float32).reshape(-1, 1)

        print(f"[Dataset] Loaded {self.observations.shape[0]} transitions.")

    def get_data(self):
        return (self.observations, self.actions, self.rewards, 
                self.next_observations, self.dones)

    def get_normalizer(self):
        obs_dim = self.observations.shape[1]
        normalizer = StateNormalizer(size=obs_dim, device=self.device)
        
        obs_tensor = torch.tensor(self.observations, device=self.device)
        normalizer.fit(obs_tensor)
        
        print(f"[Normalizer] Mean: {normalizer.mean[:5].cpu().numpy()}... ")
        print(f"[Normalizer] Std:  {normalizer.std[:5].cpu().numpy()}...")
        return normalizer