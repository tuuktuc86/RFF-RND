import torch
import torch.nn.functional as F

from .base import BaseUncertainty
from src.networks.rff_network import RNDRFFNetwork


class RNDGRFFUncertainty(BaseUncertainty):
    def __init__(self, obs_dim, action_dim, cfg, device):
        super().__init__(device)

        self.model = RNDRFFNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            lengthscale=cfg.lengthscale,
            device=device,
            predictor_conditioning=cfg.get("predictor_conditioning", "film"),
            predictor_num_layers=cfg.get("predictor_num_layers", 2),
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.predictor.parameters(), lr=cfg.lr)

    def compute_uncertainty(self, obs, action):
        pred, target = self.model(obs, action)
        error = (pred - target).pow(2).mean(dim=-1, keepdim=True)
        return torch.log10(error + 1.0)

    def update(self, batch):
        obs, action = batch["obs"], batch["action"]
        pred, target = self.model(obs, action)
        loss = F.mse_loss(pred, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"uncertainty_loss": loss.item()}

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

