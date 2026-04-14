import numpy as np
import torch
import torch.nn as nn

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FlowMatchingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
        self.embed_net = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            # nn.Linear(input_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 预测向量场 v(x, t)
        )
        self.last = nn.Sequential(
            nn.Linear(hidden_dim+time_dim, 64),
            SiLU(),
            nn.Linear(64, input_dim) ,
        )

    def forward(self, x, t, eps=1e-3):
        t = t * (1. - eps) + eps
        time_embed = self.time_embed(t.squeeze())
        xt = torch.cat([x, time_embed], dim=1)  # 连接时间维度
        xt = self.embed_net(xt)
        xt = torch.cat([xt, time_embed], dim=1)
        xt = self.last(xt)
        # xt = x
        return xt
