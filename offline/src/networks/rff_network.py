import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.base_sac_network import standard_sac_init


class ConcatFirstLayer(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(obs_dim + action_dim, hidden_dim)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return F.relu(self.proj(x))


class GatingFirstLayer(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_gate = nn.Linear(obs_dim, hidden_dim)

    def forward(self, obs, action):
        action_feat = torch.tanh(self.action_proj(action))
        gate = torch.sigmoid(self.state_gate(obs))
        return action_feat * gate


class BilinearFirstLayer(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, use_linear_terms=True):
        super().__init__()
        self.bilinear = nn.Bilinear(obs_dim, action_dim, hidden_dim, bias=True)
        self.use_linear_terms = use_linear_terms

        if use_linear_terms:
            self.obs_linear = nn.Linear(obs_dim, hidden_dim, bias=False)
            self.action_linear = nn.Linear(action_dim, hidden_dim, bias=False)

    def forward(self, obs, action):
        z = self.bilinear(obs, action)

        if self.use_linear_terms:
            z = z + self.obs_linear(obs) + self.action_linear(action)

        return F.relu(z)


class FiLMFirstLayer(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.condition = nn.Linear(obs_dim, 2 * hidden_dim)

    def forward(self, obs, action):
        h = self.action_proj(action)
        gamma_beta = self.condition(obs)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        z = (1.0 + gamma) * h + beta
        return F.relu(z)


class RNDPredictor(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        output_dim,
        conditioning="concat",
        num_layers=2,
    ):
        super().__init__()

        if conditioning == "concat":
            self.first = ConcatFirstLayer(obs_dim, action_dim, hidden_dim)
        elif conditioning == "gating":
            self.first = GatingFirstLayer(obs_dim, action_dim, hidden_dim)
        elif conditioning == "bilinear":
            self.first = BilinearFirstLayer(obs_dim, action_dim, hidden_dim)
        elif conditioning == "film":
            self.first = FiLMFirstLayer(obs_dim, action_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown predictor conditioning: {conditioning}")

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, obs, action):
        z = self.first(obs, action)
        return self.head(z)


class RNDRFFNetwork(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        output_dim,
        lengthscale,
        device,
        predictor_conditioning="concat",
        predictor_num_layers=2,
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim

        input_dim = obs_dim + action_dim
        sigma = 1.0 / lengthscale

        omega = torch.randn(output_dim, input_dim, device=device) * sigma
        b = 2.0 * np.pi * torch.rand(output_dim, device=device)

        self.register_buffer("target_omega", omega)
        self.register_buffer("target_b", b)
        self.target_scale = np.sqrt(2.0)

        self.predictor = RNDPredictor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            conditioning=predictor_conditioning,
            num_layers=predictor_num_layers,
        )
        self.predictor.apply(standard_sac_init)

        last_layer = self.predictor.head[-1]
        nn.init.uniform_(last_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(last_layer.bias, -3e-3, 3e-3)

    def forward_target(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        proj = x @ self.target_omega.t() + self.target_b[None, :]
        return self.target_scale * torch.cos(proj)

    def forward(self, obs, action):
        target_feat = self.forward_target(obs, action)
        pred_feat = self.predictor(obs, action)
        return pred_feat, target_feat
