import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)
        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            self.sample_noise()
        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnActorCriticNetwork, self).__init__()
        if use_noisy_net:
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            linear(7 * 7 * 64, 256),
            nn.ReLU(),
            linear(256, 448),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ReLU(),
            linear(448, output_size)
        )
        self.extra_layer = nn.Sequential(
            linear(448, 448),
            nn.ReLU()
        )
        self.critic_ext = linear(448, 1)
        self.critic_int = linear(448, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()
        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)
        return policy, value_ext, value_int

class RNDNetwork(nn.Module):
    def __init__(self, output_dim, device="cuda"):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        feature_output = 7 * 7 * 64

        self.predictor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        ).to(device)

        self.target = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        ).to(device)

        for p in self.target.parameters():
            p.requires_grad = False

        self._init_weights()
                        
    def _init_weights(self):
        for m in self.predictor.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    init.zeros_(m.bias)
        last_layer = self.predictor[-1]
        init.uniform_(last_layer.weight, -3e-3, 3e-3)
        init.uniform_(last_layer.bias, -3e-3, 3e-3)

    def forward(self, x):
        pred_feat = self.predictor(x)
        with torch.no_grad():
            target_feat = self.target(x)
        return pred_feat, target_feat

class RNDRFFNetwork(nn.Module):
    def __init__(self, output_dim=512, lengthscale=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        feature_output = 7 * 7 * 64

        self.predictor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        ).to(device)

        self.target_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
        ).to(device)
        
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        feature_dim = 512
        sigma = 1.0 / lengthscale
        omega = torch.randn(output_dim, feature_dim, device=device) * sigma
        b = 2.0 * np.pi * torch.rand(output_dim, device=device)

        self.register_buffer("target_omega", omega)
        self.register_buffer("target_b", b)
        self.target_scale = math.sqrt(2.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.predictor.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    init.zeros_(m.bias)
        last_layer = self.predictor[-1]
        init.uniform_(last_layer.weight, -3e-3, 3e-3)
        init.uniform_(last_layer.bias, -3e-3, 3e-3)

    def forward_phi(self, x):
        proj = x @ self.target_omega.t() + self.target_b[None, :]
        return self.target_scale * torch.cos(proj)

    def forward(self, x):
        pred_feat = self.predictor(x)
        with torch.no_grad():
            feat = self.target_encoder(x) 
            target = self.forward_phi(feat) 
        return pred_feat, target