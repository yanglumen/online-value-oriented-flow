import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.multistep_rl_flow_hyperparameter import *
from models.rl_flow_forward_process import sample_weighted_interpolated_points


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

class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim, args):
        super().__init__()
        if args.large_q0_model:
            dims = [state_dim + action_dim, 256, 512, 512, 256, 1]
        else:
            dims = [state_dim + action_dim, 256, 256, 256, 1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None, use_q1=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        if use_q1 is True:
            return self.q1(as_)
        if use_q1 is False:
            return self.q2(as_)
        return torch.min(self.q1(as_), self.q2(as_))

class ValueFunc(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)

class LargeValueFunc(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 512, 512, 256, 1]
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

class TimeValueFunc(nn.Module):
    def __init__(self, argus, input_dim, output_dim=1, hidden_dim=256, time_dim=64):
        super().__init__()
        self.argus = argus
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
        self.embed_net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            # nn.Linear(input_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 预测向量场 v(x, t)
        )
        self.last = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, 64),
            SiLU(),
            nn.Linear(64, output_dim),
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

class LargeTimeValueFunc(nn.Module):
    def __init__(self, argus, input_dim, output_dim=1, hidden_dim=256, time_dim=64):
        super().__init__()
        self.argus = argus
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
        self.embed_net1 = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            # nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_net2 = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 预测向量场 v(x, t)
        )
        self.last = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x, t, eps=1e-3):
        t = t * (1. - eps) + eps
        time_embed = self.time_embed(t.squeeze())
        if len(time_embed.shape) != len(x.shape):
            time_embed = time_embed.unsqueeze(1).repeat(1, x.shape[1], 1)
        xt = torch.cat([x, time_embed], dim=-1)
        xt = self.embed_net1(xt)
        xt = torch.cat([xt, time_embed], dim=-1)
        xt = self.embed_net2(xt)
        xt = torch.cat([xt, time_embed], dim=-1)
        xt = self.last(xt)
        return xt

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

class flow_constrained_iql_critic(nn.Module):
    def __init__(self, args, sdim, adim):
        super().__init__()
        self.discount = args.discount
        self.argus = args
        self.q = TwinQ(adim, sdim, args).to(args.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(args.device)
        if args.rl_mode in [RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
            self.v = ValueFunc(sdim).to(args.device)
            if args.large_flow_V:
                self.flow_v = LargeValueFunc(sdim+adim+adim).to(args.device)
            else:
                self.flow_v = ValueFunc(sdim+adim+adim).to(args.device)
        elif args.rl_mode == RLTrainMode.flow_constrained_rl3:
            self.v = ValueFunc(sdim).to(args.device)
            self.flow_v = ValueFunc(sdim+adim).to(args.device)
        elif args.rl_mode == RLTrainMode.flow_constrained_rl:
            self.v = ValueFunc(sdim+adim).to(args.device)
        else:
            raise NotImplementedError
        self.ema = EMA(args.ema_decay)

    def get_adv(self, observations, actions, last_actions, x_t, dx_dt, t):
        if self.argus.rl_mode in [RLTrainMode.flow_constrained_rl4]:
            return self.get_scaled_q(obs=observations, act=actions)
        else:
            raise NotImplementedError

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_target_q(self, obs, act, scale=1.0):
        return self.q_target(action=act, condition=obs)/scale

    def get_scaled_q(self, obs, act, scale=1.0):
        return self.q(action=act, condition=obs)/scale

    def get_qs(self, obs, act):
        return self.q.both(action=act, condition=obs)

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_v_loss(self, tau, observations, actions):
        with torch.no_grad():
            target_q = self.q_target(actions, observations)
        value = self.v(observations)
        v0_loss = self.expectile_loss(tau=tau, u=target_q.detach() - value)
        return v0_loss, {"v0_mean_value": value.mean().detach().cpu().numpy().item(),
                         "v0_loss": v0_loss.detach().cpu().numpy().item()}

    def expectile_q_loss(
            self, observations, actions, next_observations, rewards, dones,
            fake_next_actions=None):
        with torch.no_grad():
            next_energy = self.v(next_observations)
        targets = rewards + (1. - dones) * self.discount * next_energy.detach()
        qs = self.q.both(actions, observations)
        q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                         "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                         "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}

    def q_loss(self, observations, actions, rewards, dones):
        with torch.no_grad():
            target_qs = self.q_target.both(actions, observations)
            target_qs = torch.min(torch.cat(target_qs, dim=-1), dim=-1, keepdim=True).values
        terminal_loss = F.mse_loss(target_qs*dones, rewards*dones)
        targets = rewards[:-1] + (1. - dones[:-1]) * self.discount * target_qs[1:].detach()
        qs = self.q.both(actions[:-1], observations[:-1])
        q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs) + terminal_loss
        return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                         "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                         "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}

    def flow_constrained_rl5_loss(self, flow, behavior_flow, energy_model, observations, actions, next_observations, rewards, dones, tau, multiple_actions = 10):
        loss_info = {}

        num_samples = len(observations)
        x = torch.randn(num_samples, multiple_actions, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # x = x.unsqueeze(1).repeat(1, multiple_actions, 1)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, multiple_actions, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t=t_tensor)
                divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            # target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            # min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            target_q = energy_model.get_scaled_target_q(obs=next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), act=x, scale=self.argus.energy_scale)
            min_Q = target_q
        flow_v_loss = 0
        for step_i in range(len(multi_divergence)):
            target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            # target_flow_V = torch.mean(target_flow_V, dim=1)
            flow_V = self.flow_v(torch.cat([multi_x[step_i], actions.unsqueeze(1).repeat(1, multiple_actions, 1), next_observations.unsqueeze(1).repeat(1, multiple_actions, 1)], dim=-1))
            # flow_V = torch.mean(flow_V, dim=1)
            flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach())/len(multi_divergence)
        loss_info.update({"flow_v_loss": flow_v_loss.detach().cpu().numpy().item(), "flow_V_value": flow_V.mean().detach().cpu().numpy().item()})

        return None, None, flow_v_loss, loss_info

    def flow_constrained_rl4_loss(self, flow, behavior_flow, energy_model, observations, actions, next_observations, rewards, dones, tau, multiple_actions = 10):
        loss_info = {}
        v_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions
        )
        loss_info.update(info)
        q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=None
        )
        loss_info.update(info)
        # q_loss, info = self.q_loss(
        #     observations=observations, actions=actions, rewards=rewards, dones=dones
        # )
        # loss_info.update(info)

        num_samples = len(observations)
        x = torch.randn(num_samples, multiple_actions, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # x = x.unsqueeze(1).repeat(1, multiple_actions, 1)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, multiple_actions, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t=t_tensor)
                if self.argus.dataset in self.argus.adroit_dataset:
                    v = v.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                    divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                elif self.argus.dataset in self.argus.maze2d_dataset:
                    v = v.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                    divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                elif self.argus.dataset in self.argus.antmaze_dataset:
                    v = v.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                    divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                else:
                    divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            # target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            # min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            if self.argus.dataset == "halfcheetah-medium-expert-v2":
                target_q = energy_model.get_scaled_target_q(obs=next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), act=x, scale=self.argus.energy_scale)
            else:
                target_q = self.get_scaled_target_q(obs=next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), act=x, scale=self.argus.energy_scale)
            # target_q = self.q_target(x, next_observations.unsqueeze(1).repeat(1, multiple_actions, 1))
            min_Q = target_q
        flow_v_loss = 0
        for step_i in range(len(multi_divergence)):
            target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            # target_flow_V = torch.mean(target_flow_V, dim=1)
            flow_V = self.flow_v(torch.cat([multi_x[step_i], actions.unsqueeze(1).repeat(1, multiple_actions, 1), next_observations.unsqueeze(1).repeat(1, multiple_actions, 1)], dim=-1))
            # flow_V = torch.mean(flow_V, dim=1)
            flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach())/len(multi_divergence)
        loss_info.update({"flow_v_loss": flow_v_loss.detach().cpu().numpy().item(), "flow_V_value": flow_V.mean().detach().cpu().numpy().item()})

        return q_loss, v_loss, flow_v_loss, loss_info

    def flow_constrained_rl3_loss(self, flow, behavior_flow, energy_model, observations, actions, next_observations, rewards, dones, tau, multiple_actions = 10):
        loss_info = {}
        v_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions
        )
        loss_info.update(info)
        q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=None
        )
        loss_info.update(info)

        num_samples = len(observations)
        x = torch.randn(num_samples, multiple_actions, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, multiple_actions, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t=t_tensor)
                if self.argus.dataset in self.argus.adroit_dataset:
                    v = v.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                    divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                elif self.argus.dataset in self.argus.maze2d_dataset:
                    v = v.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                    divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                elif self.argus.dataset in self.argus.antmaze_dataset:
                    v = v.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                    divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                else:
                    divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            # target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            # min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            if self.argus.dataset == "halfcheetah-medium-expert-v2":
                target_q = energy_model.get_scaled_target_q(obs=next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), act=x, scale=self.argus.energy_scale)
            else:
                target_q = self.get_scaled_target_q(obs=next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), act=x, scale=self.argus.energy_scale)
            min_Q = target_q
        flow_v_loss = 0
        for step_i in range(len(multi_divergence)):
            target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            flow_V = self.flow_v(torch.cat([multi_x[step_i], next_observations.unsqueeze(1).repeat(1, multiple_actions, 1)], dim=-1))
            flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach())/len(multi_divergence)
        loss_info.update({"flow_v_loss": flow_v_loss.detach().cpu().numpy().item(), "flow_V_value": flow_V.mean().detach().cpu().numpy().item()})

        return q_loss, v_loss, flow_v_loss, loss_info

    def flow_constrained_rl2_loss(self, flow, behavior_flow, observations, actions, next_observations, rewards, dones, tau):
        loss_info = {}
        v_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions
        )
        loss_info.update(info)
        q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=None
        )
        loss_info.update(info)

        num_samples = len(observations)
        x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        total_divergence = 0
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations, x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations, x], dim=-1), t=t_tensor)
                divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                total_divergence += divergence
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            # target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            # min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            target_q = self.q_target(x, next_observations)
            min_Q = target_q
        flow_v_loss = 0
        for step_i in range(len(multi_divergence)):
            target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            flow_V = self.flow_v(torch.cat([multi_x[step_i], actions, next_observations], dim=-1))
            flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach())/len(multi_divergence)
        loss_info.update({"flow_v_loss": flow_v_loss.detach().cpu().numpy().item(), "flow_V_value": flow_V.mean().detach().cpu().numpy().item()})

        return q_loss, v_loss, flow_v_loss, loss_info

    def flow_constrained_rl_loss(self, flow, behavior_flow, observations, actions, next_observations, rewards, dones, tau, multiple_actions = 10):
        num_samples = len(observations)
        x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        total_divergence = 0
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations, x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations, x], dim=-1), t=t_tensor)
                divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                total_divergence += divergence
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            target_Q = rewards + (1. - dones) * self.discount * (min_Q.detach() - self.argus.divergence_coef * total_divergence)
        qs1, qs2 = self.q.both(actions, observations)
        Q_loss = 0.5*(F.mse_loss(qs1, target_Q.detach()) + F.mse_loss(qs2, target_Q.detach()))

        # random_t = np.random.uniform(low=0.0, high=1.0-dt)
        # t_tensor = torch.full((num_samples, multiple_actions, 1), t, dtype=torch.float32, device=self.argus.device)
        # x_1 = actions.unsqueeze(dim=1).repeat(1, multiple_actions, 1)
        # x_0 = torch.randn_like(x_1, device=actions.device)
        # x_0 = torch.clip(x_0, -self.argus.clip_value, self.argus.clip_value)
        # x_t = (1 - t_tensor) * x_0 + t_tensor * x_1
        # target_V = (t_tensor > 0.1).float()

        V_loss = 0
        for step_i in range(len(multi_divergence)):
            target_V = -self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q
            V_value = self.v(torch.cat([multi_x[step_i], next_observations], dim=-1))
            V_loss += F.mse_loss(V_value, target_V.detach()) / len(multi_x)
        V0 = self.v(torch.cat([x, next_observations], dim=-1))
        V_loss += F.mse_loss(V0, min_Q.detach()) / len(multi_x)

        return Q_loss, V_loss, {"q0_loss": Q_loss.detach().cpu().numpy().item(),
                                "qs1_mean_value": qs1.mean().detach().cpu().numpy().item(),
                                "qs2_mean_value": qs2.mean().detach().cpu().numpy().item(),
                                "v0_mean_value": V0.mean().detach().cpu().numpy().item(),
                                "v0_loss": V_loss.detach().cpu().numpy().item(),
                                }

class flow_constrained_ddqn_critic(nn.Module):
    def __init__(self, args, sdim, adim):
        super().__init__()
        self.discount = args.discount
        self.argus = args
        self.q = TwinQ(adim, sdim, args).to(args.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(args.device)
        if args.rl_mode in [RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
            if args.large_flow_V:
                self.flow_v = LargeValueFunc(sdim+adim+adim).to(args.device)
            else:
                self.flow_v = ValueFunc(sdim+adim+adim).to(args.device)
        elif args.rl_mode == RLTrainMode.flow_constrained_rl3:
            self.flow_v = LargeTimeValueFunc(args, sdim+adim+adim).to(args.device)
        elif args.rl_mode == RLTrainMode.flow_constrained_rl:
            pass
        else:
            raise NotImplementedError
        self.ema = EMA(args.ema_decay)

    def get_adv(self, observations, actions, last_actions, x_t, dx_dt, t):
        if self.argus.rl_mode in [RLTrainMode.flow_constrained_rl4]:
            return self.get_scaled_q(obs=observations, act=actions)
        else:
            raise NotImplementedError

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_target_q(self, obs, act, scale=1.0):
        return self.q_target(action=act, condition=obs)/scale

    def get_scaled_q(self, obs, act, scale=1.0):
        return self.q(action=act, condition=obs)/scale

    def get_qs(self, obs, act):
        return self.q.both(action=act, condition=obs)

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_v_loss(self, tau, observations, actions):
        with torch.no_grad():
            target_q = self.q_target(actions, observations)
        value = self.v(observations)
        v0_loss = self.expectile_loss(tau=tau, u=target_q.detach() - value)
        return v0_loss, {"v0_mean_value": value.mean().detach().cpu().numpy().item(),
                         "v0_loss": v0_loss.detach().cpu().numpy().item()}

    def expectile_q_loss(
            self, observations, actions, next_observations, rewards, dones,
            fake_next_actions=None):
        with torch.no_grad():
            next_energy = self.v(next_observations)
        targets = rewards + (1. - dones) * self.discount * next_energy.detach()
        qs = self.q.both(actions, observations)
        q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                         "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                         "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}

    def q_loss(self, observations, actions, rewards, dones):
        with torch.no_grad():
            target_qs = self.q_target.both(actions, observations)
            target_qs = torch.min(torch.cat(target_qs, dim=-1), dim=-1, keepdim=True)
        terminal_loss = F.mse_loss(target_qs*dones, rewards*dones)
        targets = rewards[:-1] + (1. - dones[:-1]) * self.discount * target_qs[1:].detach()
        qs = self.q.both(actions[:-1], observations[:-1])
        q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs) + terminal_loss
        return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                         "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                         "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}

    def flow_constrained_rl5_loss(self, flow, behavior_flow, energy_model, observations, actions, next_observations, rewards, dones, tau, multiple_actions = 10):
        loss_info = {}
        num_samples = len(observations)
        x = torch.randn(num_samples, multiple_actions, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # x = x.unsqueeze(1).repeat(1, multiple_actions, 1)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, multiple_actions, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t=t_tensor)
                divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            # target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            # min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            target_q = energy_model.get_scaled_target_q(obs=next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), act=x, scale=self.argus.energy_scale)
            min_Q = target_q
        flow_v_loss = 0
        for step_i in range(len(multi_divergence)):
            target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            # target_flow_V = torch.mean(target_flow_V, dim=1)
            flow_V = self.flow_v(torch.cat([multi_x[step_i], actions.unsqueeze(1).repeat(1, multiple_actions, 1), next_observations.unsqueeze(1).repeat(1, multiple_actions, 1)], dim=-1))
            # flow_V = torch.mean(flow_V, dim=1)
            flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach())/len(multi_divergence)
        loss_info.update({"flow_v_loss": flow_v_loss.detach().cpu().numpy().item(), "flow_V_value": flow_V.mean().detach().cpu().numpy().item()})

        return None, None, flow_v_loss, loss_info

    def flow_constrained_rl4_loss(self, flow, behavior_flow, energy_model, observations, actions, next_observations, rewards, dones, tau, multiple_actions = 10):
        loss_info = {}
        v_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions
        )
        loss_info.update(info)
        q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=None
        )
        loss_info.update(info)

        num_samples = len(observations)
        x = torch.randn(num_samples, multiple_actions, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # x = x.unsqueeze(1).repeat(1, multiple_actions, 1)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, multiple_actions, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), x], dim=-1), t=t_tensor)
                divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            # target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            # min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            if self.argus.dataset == "halfcheetah-medium-expert-v2":
                target_q = energy_model.get_scaled_target_q(obs=next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), act=x, scale=self.argus.energy_scale)
            else:
                target_q = self.get_scaled_target_q(obs=next_observations.unsqueeze(1).repeat(1, multiple_actions, 1), act=x, scale=self.argus.energy_scale)
            # target_q = self.q_target(x, next_observations.unsqueeze(1).repeat(1, multiple_actions, 1))
            min_Q = target_q
        flow_v_loss = 0
        for step_i in range(len(multi_divergence)):
            target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            # target_flow_V = torch.mean(target_flow_V, dim=1)
            flow_V = self.flow_v(torch.cat([multi_x[step_i], actions.unsqueeze(1).repeat(1, multiple_actions, 1), next_observations.unsqueeze(1).repeat(1, multiple_actions, 1)], dim=-1))
            # flow_V = torch.mean(flow_V, dim=1)
            flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach())/len(multi_divergence)
        loss_info.update({"flow_v_loss": flow_v_loss.detach().cpu().numpy().item(), "flow_V_value": flow_V.mean().detach().cpu().numpy().item()})

        return q_loss, v_loss, flow_v_loss, loss_info

    def flow_constrained_rl3_loss(self, flow, behavior_flow, observations, actions, next_observations, rewards, dones, tau):
        loss_info = {}
        v_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions
        )
        loss_info.update(info)
        q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=None
        )
        loss_info.update(info)

        num_samples = len(observations)
        x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        multi_t = []
        total_divergence = 0
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations, x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations, x], dim=-1), t=t_tensor)
                divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                total_divergence += divergence
                multi_divergence.append(divergence)
                multi_t.append(t_tensor)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            # target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            # min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            target_q = self.q_target(x, next_observations)
            min_Q = target_q
        flow_v_loss = 0
        for step_i in range(len(multi_divergence)):
            target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            flow_V = self.flow_v(torch.cat([multi_x[step_i], actions, next_observations], dim=-1), t=multi_t[step_i])
            flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach()) / len(multi_divergence)
        loss_info.update({"flow_v_loss": flow_v_loss.detach().cpu().numpy().item(),
                          "flow_V_value": flow_V.mean().detach().cpu().numpy().item()})

        return q_loss, v_loss, flow_v_loss, loss_info

    def flow_constrained_rl2_loss(self, flow, behavior_flow, observations, actions, next_observations, rewards, dones, tau):
        loss_info = {}
        v_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions
        )
        loss_info.update(info)
        q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=None
        )
        loss_info.update(info)

        num_samples = len(observations)
        x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        total_divergence = 0
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations, x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations, x], dim=-1), t=t_tensor)
                divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                total_divergence += divergence
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            # target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            # min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            target_q = self.q_target(x, next_observations)
            min_Q = target_q
        flow_v_loss = 0
        for step_i in range(len(multi_divergence)):
            target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            flow_V = self.flow_v(torch.cat([multi_x[step_i], actions, next_observations], dim=-1))
            flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach())/len(multi_divergence)
        loss_info.update({"flow_v_loss": flow_v_loss.detach().cpu().numpy().item(), "flow_V_value": flow_V.mean().detach().cpu().numpy().item()})

        return q_loss, v_loss, flow_v_loss, loss_info

    def flow_constrained_rl_loss(self, flow, behavior_flow, observations, actions, next_observations, rewards, dones, tau, multiple_actions = 10):
        num_samples = len(observations)
        x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        dt = 1.0 / self.argus.flow_step
        multi_x = [x]
        multi_divergence = []
        total_divergence = 0
        step_index = 0
        time_start, time_end, steps = 0, 1.0, self.argus.flow_step
        with torch.no_grad():
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = flow(torch.cat([next_observations, x], dim=-1), t_tensor)
                behavior_u = behavior_flow(x=torch.cat([next_observations, x], dim=-1), t=t_tensor)
                divergence = torch.norm(v - behavior_u, dim=-1, keepdim=True)
                total_divergence += divergence
                multi_divergence.append(divergence)
                x = x + v * dt * self.argus.flow_step_scale
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                multi_x.append(x)
                step_index += 1
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            target_qs1, target_qs2 = self.q_target.both(x, next_observations)
            min_Q = torch.min(torch.cat([target_qs1, target_qs2], dim=-1), dim=-1, keepdim=True).values
            target_Q = rewards + (1. - dones) * self.discount * (min_Q.detach() - self.argus.divergence_coef * total_divergence)
        qs1, qs2 = self.q.both(actions, observations)
        Q_loss = 0.5*(F.mse_loss(qs1, target_Q.detach()) + F.mse_loss(qs2, target_Q.detach()))

        # random_t = np.random.uniform(low=0.0, high=1.0-dt)
        # t_tensor = torch.full((num_samples, multiple_actions, 1), t, dtype=torch.float32, device=self.argus.device)
        # x_1 = actions.unsqueeze(dim=1).repeat(1, multiple_actions, 1)
        # x_0 = torch.randn_like(x_1, device=actions.device)
        # x_0 = torch.clip(x_0, -self.argus.clip_value, self.argus.clip_value)
        # x_t = (1 - t_tensor) * x_0 + t_tensor * x_1
        # target_V = (t_tensor > 0.1).float()

        V_loss = 0
        for step_i in range(len(multi_divergence)):
            target_V = -self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q
            V_value = self.v(torch.cat([multi_x[step_i], next_observations], dim=-1))
            V_loss += F.mse_loss(V_value, target_V.detach()) / len(multi_x)
        V0 = self.v(torch.cat([x, next_observations], dim=-1))
        V_loss += F.mse_loss(V0, min_Q.detach()) / len(multi_x)

        return Q_loss, V_loss, {"q0_loss": Q_loss.detach().cpu().numpy().item(),
                                "qs1_mean_value": qs1.mean().detach().cpu().numpy().item(),
                                "qs2_mean_value": qs2.mean().detach().cpu().numpy().item(),
                                "v0_mean_value": V0.mean().detach().cpu().numpy().item(),
                                "v0_loss": V_loss.detach().cpu().numpy().item(),
                                }

class direct_flow2result_iql_critic(nn.Module):
    def __init__(self, args, sdim, adim):
        super().__init__()
        self.discount = args.discount
        self.argus = args
        self.q = TwinQ(adim, sdim, args).to(args.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(args.device)
        self.v = ValueFunc(sdim).to(args.device)
        self.ema = EMA(args.ema_decay)

    def get_adv(self, observations, x_t, dx_dt, t):
        raise NotImplementedError

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_q(self, obs, act, scale=1.0):
        return self.q(action=act, condition=obs)/scale

    def get_scaled_v(self, obs, scale=1.0):
        return self.v(obs)/scale

    def get_qs(self, obs, act):
        return self.q.both(action=act, condition=obs)

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_v_loss(self, tau, observations, actions):
        with torch.no_grad():
            target_q = self.q_target(actions, observations)
        value = self.v(observations)
        v0_loss = self.expectile_loss(tau=tau, u=target_q.detach() - value)
        return v0_loss, {"v0_mean_value": value.mean().detach().cpu().numpy().item(),
                         "v0_loss": v0_loss.detach().cpu().numpy().item()}

    def expectile_q_loss(
            self, observations, actions, next_observations, rewards, dones,
            fake_next_actions=None):
        with torch.no_grad():
            next_energy = self.v(next_observations)
        targets = rewards + (1. - dones) * self.discount * next_energy.detach()
        qs = self.q.both(actions, observations)
        q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        return q0_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                         "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                         "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}

    def direct_flow2result_loss(self, observations, actions, next_observations, rewards, dones, tau, multiple_actions = 10):
        loss_info = {}
        v_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions
        )
        loss_info.update(info)
        q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=None
        )
        loss_info.update(info)

        return q_loss, v_loss, loss_info

class ciql_critic(nn.Module):

    def __init__(self, args, sdim, adim):
        super().__init__()
        self.discount = args.discount
        self.args = args
        self.q = TwinQ(adim, sdim, args).to(args.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(args.device)
        self.v = ValueFunc(sdim).to(args.device)
        self.ema = EMA(args.ema_decay)
        self.alpha = 1.0

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_q(self, obs, act, scale=1.0):
        return self.q_target(action=act, condition=obs)/scale

    def get_qs(self, obs, act):
        return self.q_target.both(action=act, condition=obs)

    def get_scaled_v(self, obs, scale=1.0):
        return self.v(state=obs)/scale

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_v_loss(self, tau, observations, actions):
        with torch.no_grad():
            target_q = self.q_target(actions, observations)
        value = self.v(observations)
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
            self, observations, actions, next_observations, rewards, dones,
            fake_next_actions=None):
        with torch.no_grad():
            next_energy = self.v(next_observations)
        targets = rewards + (1. - dones) * self.discount * next_energy.detach()
        qs = self.q.both(actions, observations)
        q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        ramdom_actions = self.generate_random_action(actions)
        qs_random_action = self.q.both(ramdom_actions, observations)
        c_loss = (torch.logsumexp(qs_random_action[0], dim=0) - qs[0].mean()) + (torch.logsumexp(qs_random_action[1], dim=0) - qs[1].mean())
        total_loss = q0_loss + self.alpha * c_loss
        return total_loss, {"q0_loss": q0_loss.detach().cpu().numpy().item(),
                            "c_loss": c_loss.detach().cpu().numpy().item(),
                            "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
                            "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()}

    def loss(
            self, tau, observations, actions, next_observations, next_actions, rewards, dones,
            fake_actions, fake_next_actions=None, coef=1.0):
        loss_info = {}
        v0_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions
        )
        loss_info.update(info)
        bellman_q_loss, info = self.expectile_q_loss(
            observations=observations, actions=actions, next_observations=next_observations,
            rewards=rewards, dones=dones, fake_next_actions=fake_next_actions
        )
        loss_info.update(info)
        q0_loss = bellman_q_loss# + conservative_loss
        return q0_loss, v0_loss, loss_info
