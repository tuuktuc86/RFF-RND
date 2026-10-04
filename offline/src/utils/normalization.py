import torch

class StateNormalizer:
    def __init__(self, size, eps=1e-6, device="cpu"):
        self.mean = torch.zeros(size, device=device)
        self.std = torch.ones(size, device=device)
        self.eps = eps
        self.device = device

    def fit(self, data: torch.Tensor):
        data = data.to(self.device)
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0) + self.eps

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std

    def denormalize(self, obs: torch.Tensor) -> torch.Tensor:
        return obs * self.std + self.mean
    
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}