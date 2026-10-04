import torch
import torch.nn as nn
import torch.distributions as td
from .blocks import mlp
import numpy as np
def standard_sac_init(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__()
        self.log_std_min = -5
        self.log_std_max = 2
        
        hidden_dims = cfg.hidden_dims 
        
       
        self.trunk = mlp(state_dim, hidden_dims, hidden_dims[-1], output_activation=nn.ReLU)
        
        self.trunk.apply(standard_sac_init)

        self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)


        nn.init.uniform_(self.mu_head.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.mu_head.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_head.bias, -1e-3, 1e-3)

    def forward(self, state):
        x = self.trunk(state)
        
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = td.Normal(mu, std)
        return dist

    def sample(self, state):
        dist = self(state)
        x = dist.rsample()
        y = torch.tanh(x)
        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - y.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return y, log_prob, torch.tanh(dist.mean)

    @torch.no_grad()
    def evaluation(self, state):
        dist = self(state)              
        action = torch.tanh(dist.mean)  
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__()
        
        hidden_dims = cfg.hidden_dims
        
        self.q1 = mlp(state_dim + action_dim, hidden_dims, 1)
        self.q2 = mlp(state_dim + action_dim, hidden_dims, 1)
        
        self.q1.apply(standard_sac_init)
        self.q2.apply(standard_sac_init)
        

        self._init_output_layer(self.q1[-1])
        self._init_output_layer(self.q2[-1])

    def _init_output_layer(self, layer):
        nn.init.uniform_(layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(layer.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    


def standard_sac_init(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module

