import numpy as np
import torch
import torch.nn as nn

def mlp(dims, activation=nn.LeakyReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class inverse_dynamics(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        dims = [input_dim, 256, 256, 256, output_dim]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)
