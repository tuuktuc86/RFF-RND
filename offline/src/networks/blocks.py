import torch
import torch.nn as nn

def mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, output_activation=None):
    layers = []
    dims = [input_dim] + list(hidden_dims)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-1], output_dim))
    if output_activation:
        layers.append(output_activation())
    return nn.Sequential(*layers)