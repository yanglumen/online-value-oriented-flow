import torch
from config.f2b_hyperparameter import WeightedSamplesType

def sample_interpolated_points(batch_size, data):
    t = torch.rand(batch_size, 1, device=data.device)  # 采样时间 t ∈ [0,1]
    x_1 = data  # 目标数据
    x_0 = torch.randn_like(x_1, device=data.device)  # 以标准高斯为起始分布

    x_t = (1 - t) * x_0 + t * x_1  # 线性插值
    dx_dt = x_1 - x_0  # 真实的流动速度
    return x_t, t, dx_dt

def sample_guided_interpolated_points(data, energy):
    perm = torch.randperm(len(data))
    dataB = data[perm]
    energyB = energy[perm]
    t = torch.rand(len(data), 1, device=data.device)
    condition = (energy < energyB).float()
    x_t = condition * (data * (1 - t) + dataB * t) + (1 - condition) * (data * t + dataB * (1 - t))
    dx_dt = condition * (dataB-data) + (1 - condition) * (data - dataB)
    return x_t, t+1.0, dx_dt

def sample_weighted_interpolated_points(
        argus, observations, actions, energy, beta, energy_model, action_imitation_model, weighted_samples_type,
        time_offset=0, x_0=None, next_observations=None, next_energy=None):
    if weighted_samples_type == WeightedSamplesType.subopt2opt:
        generate_actions = action_imitation_model.gen_action(states=observations, steps=10)
        permed_energy = energy_model.get_scaled_q(obs=observations, act=generate_actions, scale=argus.energy_scale)
        t = torch.rand(len(actions), 1, device=actions.device)
        condition = (energy < permed_energy).float()
        x_t = condition * (actions * (1 - t) + generate_actions * t) + (1 - condition) * (actions * t + generate_actions * (1 - t))
        dx_dt = condition * (generate_actions - actions) + (1 - condition) * (actions - generate_actions)
        weights = condition * (torch.exp(beta * permed_energy) - torch.exp(beta * energy)) + (1 - condition) * (torch.exp(beta * energy) - torch.exp(beta * permed_energy))
        # weights = condition * (torch.exp(beta * (energyB - energy))) + (1 - condition) * (torch.exp(beta * (energy - energyB)))
        return torch.cat([observations, x_t], dim=-1), t+time_offset, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.noise_action:
        t = torch.rand(len(actions), 1, device=actions.device)
        x_1 = actions
        x_0 = torch.randn_like(x_1, device=actions.device)
        permed_actions = (1 - t) * x_0 + t * x_1
        permed_energy = energy_model.get_scaled_q(obs=observations, act=permed_actions, scale=argus.energy_scale)
        condition = (energy < permed_energy).float()
        x_t = condition * (actions * (1 - t) + permed_actions * t) + (1 - condition) * (actions * t + permed_actions * (1 - t))
        # weights = condition * (torch.exp(beta * permed_energy) - torch.exp(beta * energy)) + (1 - condition) * (torch.exp(beta * energy) - torch.exp(beta * permed_energy))
        # weights = condition * (beta * x_0_energy - beta * energy) + (1 - condition) * (beta * energy - beta * x_0_energy)
        weights = condition * (torch.exp(beta * (permed_energy - energy))) + (1 - condition) * (torch.exp(beta * (energy - permed_energy)))
        dx_dt = condition * (permed_actions - actions) + (1 - condition) * (actions - permed_actions)
        return torch.cat([observations, x_t], dim=-1), t + time_offset, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.noise_interpolation:
        t = torch.rand(len(actions), 1, device=actions.device)
        x_1 = actions
        x_0 = torch.randn_like(x_1, device=actions.device)
        x_t = (1 - t) * x_0 + t * x_1
        x_0_energy = energy_model.get_scaled_q(obs=observations, act=x_0, scale=argus.energy_scale)
        condition = (energy < x_0_energy).float()
        weights = condition * (torch.exp(beta * x_0_energy) - torch.exp(beta * energy)) + (1 - condition) * (torch.exp(beta * energy) - torch.exp(beta * x_0_energy))
        # weights = condition * (beta * x_0_energy - beta * energy) + (1 - condition) * (beta * energy - beta * x_0_energy)
        # weights = condition * (torch.exp(beta * (x_0_energy - energy))) + (1 - condition) * (torch.exp(beta * (energy - x_0_energy)))
        dx_dt = condition * (x_0 - x_1) + (1 - condition) * (x_1 - x_0)
        return torch.cat([observations, x_t], dim=-1), t, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.linear_interpolation:
        t = torch.rand(len(actions), 1, device=actions.device)
        x_1 = actions
        x_0 = torch.randn_like(x_1, device=actions.device)
        x_t = (1 - t) * x_0 + t * x_1
        dx_dt = x_1 - x_0
        weights = torch.ones_like(t, device=actions.device)
        return torch.cat([observations, x_t], dim=-1), t, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.specific_obs_x0x1:
        t = torch.rand(len(observations), 1, device=observations.device)
        x_1 = observations
        x_0 = x_0
        energy_1 = energy_model.get_scaled_v(obs=x_1, scale=argus.energy_scale)
        energy_0 = energy_model.get_scaled_v(obs=x_0, scale=argus.energy_scale)
        condition = (energy_0 < energy_1).float()
        x_t = condition * (x_0 * (1 - t) + x_1 * t) + (1 - condition) * (x_0 * t + x_1 * (1 - t))
        dx_dt = condition * (x_1 - x_0) + (1 - condition) * (x_0 - x_1)
        weights = torch.ones_like(t, device=observations.device)
        cat_x_t = condition * torch.cat([x_0, x_t], dim=-1) + (1 - condition) * torch.cat([x_1, x_t], dim=-1)
        return cat_x_t, t, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.obs_linear_interpolation:
        t = torch.rand(len(next_observations), 1, device=next_observations.device)
        x_1 = next_observations
        x_0 = torch.randn_like(x_1, device=next_observations.device)
        x_t = (1 - t) * x_0 + t * x_1
        dx_dt = x_1 - x_0
        weights = torch.ones_like(t, device=next_observations.device)
        return torch.cat([observations, x_t], dim=-1), t, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.subopt_state2opt_state:
        # perm = torch.randperm(observations.size(0))
        # observations_shuffled = observations[perm]
        # energy_shuffled = energy[perm]
        # x_0 = torch.randn_like(next_observations, device=next_observations.device)
        # t = torch.rand(len(next_observations), 1, device=next_observations.device)
        # condition = (energy < energy_shuffled).float()
        # x_1 = condition * observations_shuffled + (1 - condition) * observations
        # x_t = (1 - t) * x_0 + t * x_1
        # dx_dt = x_1 - x_0
        # prefix_x_t = condition * observations + (1 - condition) * observations_shuffled
        # weights = torch.ones_like(t, device=observations.device)
        # return torch.cat([prefix_x_t, x_t], dim=-1), t, dx_dt, weights
        x_0 = torch.randn_like(next_observations, device=next_observations.device)
        t = torch.rand(len(observations), 1, device=actions.device)
        condition = (energy+1.5 < next_energy).float()
        x_1 = condition * next_observations + (1 - condition) * observations
        x_t = (1 - t) * x_0 + t * x_1
        dx_dt = x_1 - x_0
        prefix_x_t = condition * observations + (1 - condition) * next_observations
        weights = condition
        return torch.cat([prefix_x_t, x_t], dim=-1), t, dx_dt, weights

    raise NotImplementedError
