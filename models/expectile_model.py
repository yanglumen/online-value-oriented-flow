import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)


def mlp(dims, activation=nn.ReLU, output_activation=None):
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

class expectile_Q(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        dims = [input_dim, 256, 256, 256, output_dim]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)

class expectile_V(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        dims = [input_dim, 256, 256, 256, output_dim]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.register_buffer("W", torch.randn(embed_dim // 2) * scale)
    # self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class expectile_time_func(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=256, time_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
        self.embed_net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.last = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, 64),
            SiLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x, t, eps=1e-3):
        t = t * (1. - eps) + eps
        time_embed = self.time_embed(t.squeeze())
        xt = torch.cat([x, time_embed], dim=-1)
        xt = self.embed_net(xt)
        xt = torch.cat([xt, time_embed], dim=-1)
        xt = self.last(xt)
        return xt