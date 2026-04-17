import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



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

class iql_critic(nn.Module):
    def __init__(self, args, sdim, adim):
        super().__init__()
        self.discount = args.discount
        self.args = args
        self.q = TwinQ(adim, sdim, args).to(args.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(args.device)
        self.v = ValueFunc(sdim).to(args.device)
        self.ema = EMA(args.ema_decay)

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_target_q(self, obs, act, scale=1.0):
        return torch.min(*self.q_target.both(action=act, condition=obs)) / scale

    def get_scaled_q(self, obs, act, scale=1.0):
        return torch.min(*self.q.both(action=act, condition=obs)) / scale

    def get_scaled_v(self, obs, scale=1.0):
        return self.v(state=obs)/scale

    def get_scaled_min_qs(self, obs, act, scale=1.0):
        qs = self.q.both(action=act, condition=obs)
        return torch.min(*qs) / scale

    def get_qs(self, obs, act):
        return self.q.both(action=act, condition=obs)

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_v_loss(self, tau, observations, actions):
        with torch.no_grad():
            target_q = torch.min(*self.q_target.both(actions, observations))
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

    def loss(
            self, tau, observations, actions, next_observations, next_actions, rewards, dones,
            fake_actions, fake_next_actions=None, coef=1.0, **kwargs):
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

    def get_scaled_target_q(self, obs, act, scale=1.0):
        return torch.min(*self.q_target.both(action=act, condition=obs)) / scale

    def get_scaled_q(self, obs, act, scale=1.0):
        return torch.min(*self.q.both(action=act, condition=obs)) / scale

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
            target_q = torch.min(*self.q_target.both(actions, observations))
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
            fake_actions, fake_next_actions=None, coef=1.0, **kwargs):
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

class in_support_softmax_q_learning_critic(nn.Module):
    def __init__(self, args, sdim, adim):
        super().__init__()
        self.discount = args.discount
        self.args = args
        self.q = TwinQ(adim, sdim, args).to(args.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(args.device)
        self.ema = EMA(args.ema_decay)

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_scaled_target_q(self, obs, act, scale=1.0):
        return torch.min(*self.q_target.both(action=act, condition=obs)) / scale

    def get_scaled_q(self, obs, act, scale=1.0, use_q1=None):
        if use_q1 is None:
            return torch.min(*self.q.both(action=act, condition=obs)) / scale
        return self.q(action=act, condition=obs, use_q1=use_q1)/scale

    def get_qs(self, obs, act):
        return self.q_target.both(action=act, condition=obs)

    def loss(self, behavior_model, observations, actions, next_observations, rewards, dones, fake_actions, fake_next_actions, coef=1.0, **kwargs):
        loss_info = {}

        if fake_next_actions is None:
            fake_next_actions = behavior_model.behavior_action(
                states=torch.stack([next_observations] * self.args.isql_sofrmax_action_num, axis=1), steps=self.args.flow_step, x_t_clip_value=self.args.x_t_clip_value)

        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            next_qs = self.q_target.both(fake_next_actions, torch.stack([next_observations] * fake_next_actions.shape[1], axis=1))
            next_q = torch.min(*next_qs).detach().squeeze()
            next_v = torch.sum(softmax(self.args.isql_alpha * next_q) * next_q, dim=-1, keepdim=True)

        targets = rewards + (1. - dones.float()) * self.discount * next_v.detach()
        qs = self.q.both(actions, observations)
        q0_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        loss_info.update({
            "q0_loss": q0_loss.detach().cpu().numpy().item(),
            "qs1_mean_value": qs[0].mean().detach().cpu().numpy().item(),
            "qs2_mean_value": qs[1].mean().detach().cpu().numpy().item()})

        return q0_loss, None, loss_info
