import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)

class EnergyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.Linear(input_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, 1)  # 预测向量场 v(x, t)
        )

    def forward(self, x):
        return self.net(x)

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

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.register_buffer("W", torch.randn(embed_dim // 2) * scale)
    # self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=True)
  def forward(self, x):
    x_proj = x[..., None] * self.W.detach()[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class flow_value_func(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=256, time_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
        self.embed_net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.last = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, 128),
            SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, t, eps=1e-3):
        t = t * (1. - eps) + eps
        time_embed = self.time_embed(t.squeeze())
        if len(time_embed.shape) != len(x.shape):
            time_embed = time_embed.unsqueeze(1).repeat(1, x.shape[1], 1)
        xt = torch.cat([x, time_embed], dim=-1)
        xt = self.embed_net(xt)
        xt = torch.cat([xt, time_embed], dim=-1)
        xt = self.last(xt)
        return xt

class TwinQ(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=256, time_dim=64):
        super().__init__()
        self.q1 = flow_value_func(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, time_dim=time_dim)
        self.q2 = flow_value_func(input_dim, output_dim=output_dim, hidden_dim=hidden_dim, time_dim=time_dim)

    def both(self, x, t):
        return self.q1(x, t=t), self.q2(x, t=t)

    def forward(self, x, t, use_q1=True):
        if use_q1:
            return self.q1(x, t=t)
        else:
            return self.q2(x, t=t)

class ValueFunc(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)

class iql_flow_critic(nn.Module):
    def __init__(self, args, sdim, adim, use_TwinQ=False):
        super().__init__()
        self.discount = args.discount
        self.args = args
        self.use_TwinQ = use_TwinQ
        if use_TwinQ:
            self.q = TwinQ(input_dim=adim+sdim).to(args.device)
            self.q_target = TwinQ(input_dim=adim+sdim).requires_grad_(False).to(args.device)
        else:
            self.q = flow_value_func(input_dim=sdim+adim).to(args.device)
            self.q_target = flow_value_func(input_dim=sdim+adim).requires_grad_(False).to(args.device)
        self.v = flow_value_func(input_dim=sdim+adim).to(args.device)
        self.ema = EMA(args.ema_decay)

    def get_adv(self, observations, x_t, dx_dt, t):
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        Q = self.q(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t)
        V = self.v(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
        return Q-V

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_q(self, obs, act, t, scale=1.0):
        oa = torch.cat((obs, act), dim=-1)
        return self.q(oa, t=t)/scale

    def get_qs(self, obs, act, t):
        oa = torch.cat((obs, act), dim=-1)
        return self.q.both(oa, t=t)

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_v_loss(self, tau, observations, actions, normed_actions, t):
        oa = torch.cat((observations, actions), dim=-1)
        onorma = torch.cat((observations, normed_actions), dim=-1)
        with torch.no_grad():
            target_q = self.q_target(oa, t=t)
        value = self.v(onorma, t=t)
        v0_loss = self.expectile_loss(tau=tau, u=target_q.detach() - value)
        return v0_loss, {"v0_mean_value": value.mean().detach().cpu().numpy().item(),
                         "v0_loss": v0_loss.detach().cpu().numpy().item()}

    def expectile_q_loss(
            self, observations, actions, normed_actions, next_observations, rewards, dones, t, fake_next_actions=None):
        onorma = torch.cat((observations, normed_actions), dim=-1)
        with torch.no_grad():
            next_energy = self.v(onorma, t=t)
        targets = rewards + (1. - dones) * self.discount * next_energy.detach()
        oa = torch.cat((observations, actions), dim=-1)
        if self.use_TwinQ:
            qs = self.q.both(oa, t=t)
            q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
            return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                             "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                             "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}
        else:
            q = self.q(oa, t=t)
            q0_loss = F.mse_loss(q, targets)
            return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                             "qs1_mean_value": q.mean().detach().cpu().numpy().item()}

    def loss(
            self, tau, observations, actions, normed_actions, next_observations, rewards, dones, t, next_actions,
            fake_actions, fake_next_actions=None, coef=1.0):
        loss_info = {}
        v0_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions, normed_actions=normed_actions, t=t
        )
        loss_info.update(info)
        bellman_q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, normed_actions=normed_actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=fake_next_actions, t=t,
        )
        loss_info.update(info)
        q0_loss = bellman_q_loss# + conservative_loss
        return q0_loss, v0_loss, loss_info


class adv_decision_flow_iql_flow_critic(nn.Module):
    def __init__(self, args, sdim, adim, use_TwinQ=False):
        super().__init__()
        self.discount = args.discount
        self.args = args
        self.use_TwinQ = use_TwinQ
        if use_TwinQ:
            self.q = TwinQ(input_dim=adim+sdim).to(args.device)
            self.q_target = TwinQ(input_dim=adim+sdim).requires_grad_(False).to(args.device)
        else:
            self.q = flow_value_func(input_dim=sdim+adim).to(args.device)
            self.q_target = flow_value_func(input_dim=sdim+adim).requires_grad_(False).to(args.device)
        self.v = flow_value_func(input_dim=sdim+adim).to(args.device)
        self.ema = EMA(args.ema_decay)

    def get_adv(self, observations, x_t, dx_dt, t):
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        Q = self.q(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t)
        V = self.v(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
        return Q-V

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_q(self, obs, act, t, scale=1.0):
        oa = torch.cat((obs, act), dim=-1)
        return self.q(oa, t=t)/scale

    def get_scaled_target_q(self, obs, act, t, scale=1.0):
        oa = torch.cat((obs, act), dim=-1)
        return self.q_target(oa, t=t)/scale

    def get_scaled_v(self, obs, act, t, scale=1.0):
        oa = torch.cat((obs, act), dim=-1)
        return self.v(oa, t=t)/scale

    def get_qs(self, obs, act, t):
        oa = torch.cat((obs, act), dim=-1)
        return self.q.both(oa, t=t)

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_v_loss(self, tau, observations, actions, normed_actions, t):
        oa = torch.cat((observations, actions), dim=-1)
        onorma = torch.cat((observations, normed_actions), dim=-1)
        with torch.no_grad():
            target_q = self.q_target(oa, t=t)
        value = self.v(onorma, t=t)
        v0_loss = self.expectile_loss(tau=tau, u=target_q.detach() - value)
        return v0_loss, {"v0_mean_value": value.mean().detach().cpu().numpy().item(),
                         "v0_loss": v0_loss.detach().cpu().numpy().item()}

    def expectile_q_loss(
            self, observations, actions, normed_actions, next_observations, rewards, dones, t, fake_next_actions=None):
        onorma = torch.cat((observations, normed_actions), dim=-1)
        with torch.no_grad():
            next_energy = self.v(onorma, t=t)
        targets = rewards + (1. - dones) * self.discount * next_energy.detach()
        oa = torch.cat((observations, actions), dim=-1)
        if self.use_TwinQ:
            qs = self.q.both(oa, t=t)
            q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
            return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                             "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                             "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}
        else:
            q = self.q(oa, t=t)
            q0_loss = F.mse_loss(q, targets)
            return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                             "qs1_mean_value": q.mean().detach().cpu().numpy().item()}

    def loss(
            self, tau, observations, actions, normed_actions, next_observations, rewards, dones, t, next_actions,
            fake_actions, fake_next_actions=None, coef=1.0):
        loss_info = {}
        v0_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions, normed_actions=normed_actions, t=t
        )
        loss_info.update(info)
        bellman_q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, normed_actions=normed_actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=fake_next_actions, t=t,
        )
        loss_info.update(info)
        q0_loss = bellman_q_loss# + conservative_loss
        return q0_loss, v0_loss, loss_info

class ciql_flow_critic(nn.Module):
    def __init__(self, args, sdim, adim, use_TwinQ=False):
        super().__init__()
        self.discount = args.discount
        self.args = args
        self.alpha = 1.0
        self.use_TwinQ = use_TwinQ
        if use_TwinQ:
            self.q = TwinQ(input_dim=adim+sdim).to(args.device)
            self.q_target = TwinQ(input_dim=adim+sdim).requires_grad_(False).to(args.device)
        else:
            self.q = flow_value_func(input_dim=sdim+adim).to(args.device)
            self.q_target = flow_value_func(input_dim=sdim+adim).requires_grad_(False).to(args.device)
        self.v = flow_value_func(input_dim=sdim).to(args.device)
        self.ema = EMA(args.ema_decay)

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_q(self, obs, act, t, scale=1.0):
        oa = torch.cat((obs, act), dim=-1)
        return self.q(oa, t=t)/scale

    def get_qs(self, obs, act, t):
        oa = torch.cat((obs, act), dim=-1)
        return self.q.both(oa, t=t)

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_v_loss(self, tau, observations, actions, t):
        oa = torch.cat((observations, actions), dim=-1)
        with torch.no_grad():
            target_q = self.q_target(oa, t=t)
        value = self.v(observations, t=t)
        v0_loss = self.expectile_loss(tau=tau, u=target_q.detach() - value)
        return v0_loss, {"v0_mean_value": value.mean().detach().cpu().numpy().item(),
                         "v0_loss": v0_loss.detach().cpu().numpy().item()}

    def generate_random_action(self, actions):
        t = torch.rand(len(actions), 1, device=actions.device)
        x_1 = actions
        x_0 = torch.randn_like(x_1, device=actions.device)
        x_t = (1 - t) * x_0 + t * x_1
        return x_t

    def expectile_q_loss(
            self, observations, actions, next_observations, rewards, dones, t,
            fake_next_actions=None):
        with torch.no_grad():
            next_energy = self.v(next_observations, t=t)
        targets = rewards + (1. - dones) * self.discount * next_energy.detach()
        oa = torch.cat((observations, actions), dim=-1)
        if self.use_TwinQ:
            qs = self.q.both(oa, t=t)
            q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
            ramdom_actions = self.generate_random_action(actions)
            qs_random_action = self.q.both(torch.cat((observations, ramdom_actions), dim=-1))
            c_loss = (torch.logsumexp(qs_random_action[0], dim=0) - qs[0].mean()) + (torch.logsumexp(qs_random_action[1], dim=0) - qs[1].mean())
            total_loss = q0_loss + self.alpha * c_loss
            return total_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                                "c_loss": c_loss.detach().cpu().numpy().item(),
                                "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                                "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}
        else:
            q = self.q(oa, t=t)
            q0_loss = F.mse_loss(q, targets)
            ramdom_actions = self.generate_random_action(actions)
            qs_random_action = self.q(torch.cat((observations, ramdom_actions), dim=-1))
            c_loss = torch.logsumexp(qs_random_action, dim=0) - q.mean()
            total_loss = q0_loss + self.alpha * c_loss
            return total_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                                "c_loss": c_loss.detach().cpu().numpy().item(),
                                "qs1_mean_value": q.mean().detach().cpu().numpy().item()}

    def loss(
            self, tau, observations, actions, next_observations, next_actions, rewards, dones,
            fake_actions, t, fake_next_actions=None, coef=1.0):
        loss_info = {}
        v0_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions, t=t
        )
        loss_info.update(info)
        bellman_q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=fake_next_actions, t=t,
        )
        loss_info.update(info)
        q0_loss = bellman_q_loss# + conservative_loss
        return q0_loss, v0_loss, loss_info
