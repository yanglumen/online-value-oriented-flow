"""Sliding-window version of the IQL critic for near drop-in replacement of models.energy_model.iql_critic."""

import copy
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, "MLP requires at least two dims (input and output)"

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
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
        if getattr(args, "large_q0_model", False):
            dims = [state_dim + action_dim, 256, 512, 512, 256, 1]
        else:
            dims = [state_dim + action_dim, 256, 256, 256, 1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None, use_q1=True):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        if use_q1:
            return self.q1(as_)
        return torch.min(*self.both(action, condition))


class ValueFunc(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)


class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def circular_window(start: int, length: int, total: int) -> List[int]:
    if total <= 0:
        return []
    length = max(1, min(length, total))
    return [int((start + offset) % total) for offset in range(length)]


class SlidingWindowTwinQEnsemble(nn.Module):
    """Container for a ring ensemble of TwinQ heads.

    `forward(action, condition)` returns the conservative aggregate:
    min over twin heads inside each member, then min across ensemble members.
    """

    def __init__(self, action_dim, state_dim, args, num_members):
        super().__init__()
        self.num_members = int(num_members)
        self.heads = nn.ModuleList([TwinQ(action_dim, state_dim, args) for _ in range(self.num_members)])

    def both_member(self, idx, action, condition=None):
        return self.heads[idx].both(action=action, condition=condition)

    def min_member(self, idx, action, condition=None):
        q1, q2 = self.both_member(idx=idx, action=action, condition=condition)
        return torch.min(q1, q2)

    def all_member_mins(self, action, condition=None):
        mins = [self.min_member(i, action=action, condition=condition) for i in range(self.num_members)]
        return torch.stack(mins, dim=0)

    def subset_member_mins(self, indices: Sequence[int], action, condition=None):
        if len(indices) == 0:
            raise ValueError("indices must be non-empty")
        mins = [self.min_member(int(i), action=action, condition=condition) for i in indices]
        return torch.stack(mins, dim=0)

    def forward(self, action, condition=None, use_q1=True):
        del use_q1
        member_mins = self.all_member_mins(action=action, condition=condition)
        return torch.min(member_mins, dim=0).values


class sliding_window_iql_critic(nn.Module):
    """Engineering adaptation of SWDG for the repo's IQL-style energy model.

    Compared with the original `iql_critic`, this class keeps:
    - a single shared value network `self.v`
    - `self.q`, `self.q_target`, `self.v` as nn.Module attributes

    and replaces the single TwinQ with a ring ensemble of TwinQ members.

    `get_qs(obs, act)` returns a tensor of shape [N, B, 1], where N is the
    number of ensemble members and each slice is that member's twin-min Q.
    """

    def __init__(self, args, sdim, adim):
        super().__init__()
        self.args = args
        self.discount = args.discount
        self.device = getattr(args, "device", "cpu")

        self.num_q_ensembles = int(getattr(args, "swdg_num_q_ensembles", 8))
        self.window_size = int(getattr(args, "swdg_window_size", 4))
        self.window_step = int(getattr(args, "swdg_window_step", 1))
        self.use_diversity_reg = bool(getattr(args, "swdg_use_diversity_reg", False))
        self.diversity_coef = float(getattr(args, "swdg_diversity_coef", 0.0))

        self.q = SlidingWindowTwinQEnsemble(adim, sdim, args, self.num_q_ensembles).to(self.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(self.device)
        self.v = ValueFunc(sdim).to(self.device)
        self.ema = EMA(getattr(args, "ema_decay", 0.995))

        self.window_start = 0
        self.prev_active_indices: Optional[List[int]] = None
        self._last_active_indices: List[int] = self.get_active_indices()
        self._last_delayed_indices: List[int] = list(self._last_active_indices)
        self._loss_call_count = 0

    def q_ema(self):
        self.ema.update_model_average(self.q_target, self.q)

    def get_active_indices(self):
        return circular_window(self.window_start, self.window_size, self.num_q_ensembles)

    def get_delayed_indices(self, active_indices: Optional[Sequence[int]] = None):
        if active_indices is None:
            active_indices = self.get_active_indices()
        active_indices = [int(i) for i in active_indices]
        if self.prev_active_indices is None:
            # First loss call: fall back to current window for a stable, interpretable bootstrap.
            return list(active_indices)
        prev_set = set(int(i) for i in self.prev_active_indices)
        delayed = [idx for idx in active_indices if idx not in prev_set]
        if len(delayed) == 0:
            delayed = list(active_indices[-min(self.window_step, len(active_indices)):])
        return delayed

    def maybe_advance_window(self):
        self.window_start = (self.window_start + self.window_step) % self.num_q_ensembles

    def get_scaled_target_q(self, obs, act, scale=1.0):
        return self.q_target(action=act, condition=obs) / scale

    def get_scaled_q(self, obs, act, scale=1.0):
        return self.q(action=act, condition=obs) / scale

    def get_scaled_v(self, obs, scale=1.0):
        return self.v(state=obs) / scale

    def get_scaled_min_qs(self, obs, act, scale=1.0):
        return torch.min(self.q.all_member_mins(action=act, condition=obs), dim=0).values / scale

    def get_qs(self, obs, act):
        return self.q.all_member_mins(action=act, condition=obs)

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        return out.mean()

    def _delayed_target_q(self, observations, actions, delayed_indices):
        delayed_qs = self.q_target.subset_member_mins(indices=delayed_indices, action=actions, condition=observations)
        delayed_q_min = torch.min(delayed_qs, dim=0).values
        return delayed_q_min, delayed_qs

    def expectile_v_loss(self, tau, observations, actions, delayed_indices=None):
        if delayed_indices is None:
            delayed_indices = list(self._last_delayed_indices)
        with torch.no_grad():
            delayed_q_min, delayed_qs = self._delayed_target_q(
                observations=observations, actions=actions, delayed_indices=delayed_indices
            )
        value = self.v(observations)
        v0_loss = self.expectile_loss(tau=tau, u=delayed_q_min.detach() - value)
        return v0_loss, {
            "v0_mean_value": value.mean().detach().cpu().numpy().item(),
            "v0_loss": v0_loss.detach().cpu().numpy().item(),
            "delayed_q_min": delayed_q_min.mean().detach().cpu().numpy().item(),
            "delayed_q_std": delayed_qs.std(unbiased=False).detach().cpu().numpy().item(),
        }

    def _active_q_loss(self, observations, actions, targets, active_indices):
        member_losses = []
        member_means = []
        for idx in active_indices:
            q1, q2 = self.q.both_member(idx=idx, action=actions, condition=observations)
            member_loss = 0.5 * (F.mse_loss(q1, targets) + F.mse_loss(q2, targets))
            member_losses.append(member_loss)
            member_means.append(torch.min(q1, q2).mean().detach())
        q0_loss = torch.stack(member_losses).mean()
        ensemble_q_mean = torch.stack(member_means).mean().cpu().numpy().item()
        ensemble_q_std = torch.stack(member_means).std(unbiased=False).cpu().numpy().item()
        return q0_loss, ensemble_q_mean, ensemble_q_std

    def _diversity_regularization(self, observations, actions, active_indices):
        # Optional engineering extension: encourage active critics to disagree in action-gradients.
        if not self.use_diversity_reg or self.diversity_coef <= 0.0 or len(active_indices) < 2:
            return None

        act_var = actions.detach().clone().requires_grad_(True)
        grads = []
        for idx in active_indices:
            q_val = self.q.min_member(idx=idx, action=act_var, condition=observations).sum()
            grad = torch.autograd.grad(q_val, act_var, retain_graph=True, create_graph=True)[0]
            grad = grad.reshape(grad.shape[0], -1)
            grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)
            grads.append(grad)

        penalties = []
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                cosine = (grads[i] * grads[j]).sum(dim=-1, keepdim=True)
                penalties.append(cosine.mean())
        if len(penalties) == 0:
            return None
        return torch.stack(penalties).mean()

    def expectile_q_loss(
            self, observations, actions, next_observations, rewards, dones,
            fake_next_actions=None, active_indices=None):
        del fake_next_actions
        if active_indices is None:
            active_indices = list(self._last_active_indices)
        with torch.no_grad():
            next_energy = self.v(next_observations)
        targets = rewards + (1. - dones) * self.discount * next_energy.detach()
        q0_loss, ensemble_q_mean, ensemble_q_std = self._active_q_loss(
            observations=observations, actions=actions, targets=targets, active_indices=active_indices
        )

        diversity_loss = self._diversity_regularization(
            observations=observations, actions=actions, active_indices=active_indices
        )
        total_q_loss = q0_loss
        loss_info = {
            "q0_loss": q0_loss.detach().cpu().numpy().item(),
            "ensemble_q_mean": ensemble_q_mean,
            "ensemble_q_std": ensemble_q_std,
        }
        if diversity_loss is not None:
            total_q_loss = total_q_loss + self.diversity_coef * diversity_loss
            loss_info["diversity_loss"] = diversity_loss.detach().cpu().numpy().item()
        return total_q_loss, loss_info

    def loss(
            self, tau, observations, actions, next_observations, next_actions, rewards, dones,
            fake_actions, fake_next_actions=None, coef=1.0, **kwargs):
        del next_actions, fake_actions, coef, kwargs

        active_indices = self.get_active_indices()
        delayed_indices = self.get_delayed_indices(active_indices=active_indices)
        self._last_active_indices = list(active_indices)
        self._last_delayed_indices = list(delayed_indices)

        loss_info = {}
        v0_loss, info = self.expectile_v_loss(
            tau=tau, observations=observations, actions=actions, delayed_indices=delayed_indices
        )
        loss_info.update(info)

        bellman_q_loss, info = self.expectile_q_loss(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            dones=dones,
            fake_next_actions=fake_next_actions,
            active_indices=active_indices,
        )
        loss_info.update(info)

        with torch.no_grad():
            ensemble_qs = self.q.all_member_mins(action=actions, condition=observations)
            delayed_q_min = self.q_target.subset_member_mins(
                indices=delayed_indices, action=actions, condition=observations
            ).min(dim=0).values

        loss_info.update({
            "active_indices": list(active_indices),
            "delayed_indices": list(delayed_indices),
            "window_start": int(self.window_start),
            "ensemble_q_mean": ensemble_qs.mean().detach().cpu().numpy().item(),
            "ensemble_q_std": ensemble_qs.std(unbiased=False).detach().cpu().numpy().item(),
            "delayed_q_min": delayed_q_min.mean().detach().cpu().numpy().item(),
        })

        self.prev_active_indices = list(active_indices)
        self._loss_call_count += 1
        self.maybe_advance_window()
        q0_loss = bellman_q_loss
        return q0_loss, v0_loss, loss_info
