import abc
import torch

class BaseAlgorithm(abc.ABC):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    @abc.abstractmethod
    def update(self, batch: dict) -> dict:

        pass
    
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass