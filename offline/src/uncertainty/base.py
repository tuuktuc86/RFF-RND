import abc
import torch
import torch.nn as nn
class BaseUncertainty(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @abc.abstractmethod
    def compute_uncertainty(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        pass

    @abc.abstractmethod
    def update(self, batch: dict) -> dict:
 
        pass

    @abc.abstractmethod
    def state_dict(self) -> dict:
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass