import torch

class GPUReplayBuffer:
    def __init__(self, 
                 observations, 
                 actions, 
                 rewards, 
                 next_observations, 
                 dones, 
                 device):
        self.device = device
        
        self.observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1, 1)
        self.next_observations = torch.as_tensor(next_observations, dtype=torch.float32, device=device)
        self.dones = torch.as_tensor(dones, dtype=torch.float32, device=device).reshape(-1, 1)
        
        self.size = self.observations.shape[0]

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            'obs': self.observations[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices],
            'next_obs': self.next_observations[indices],
            'done': self.dones[indices]
        }