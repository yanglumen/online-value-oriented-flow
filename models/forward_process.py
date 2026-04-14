import torch
from config.hyperparameter import WeightedSamplesType

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
        argus, observations, actions, energy, beta, energy_model, weighted_samples_type, time_offset=0):
    if weighted_samples_type == WeightedSamplesType.subopt2opt:
        perm = torch.randperm(len(actions))
        permed_actions = actions[perm]
        permed_energy = energy_model.get_scaled_q(obs=observations, act=permed_actions, scale=argus.energy_scale)
        t = torch.rand(len(actions), 1, device=actions.device)
        condition = (energy < permed_energy).float()
        x_t = condition * (actions * (1 - t) + permed_actions * t) + (1 - condition) * (actions * t + permed_actions * (1 - t))
        dx_dt = condition * (permed_actions - actions) + (1 - condition) * (actions - permed_actions)
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
        # random_noise = torch.randn_like(actions, device=actions.device) * 0.1
        # permed_actions = (actions + random_noise).clip(min=-1.0, max=1.0)
        # permed_energy = energy_model.get_scaled_q(obs=observations, act=permed_actions, scale=argus.energy_scale)
        # t = torch.rand(len(actions), 1, device=actions.device)
        # condition = (energy < permed_energy).float()
        # x_t = condition * (actions * (1 - t) + permed_actions * t) + (1 - condition) * (actions * t + permed_actions * (1 - t))
        # dx_dt = condition * (permed_actions - actions) + (1 - condition) * (actions - permed_actions)
        # weights = condition * (torch.exp(beta * permed_energy) - torch.exp(beta * energy)) + (1 - condition) * (torch.exp(beta * energy) - torch.exp(beta * permed_energy))
        # # weights = condition * (torch.exp(beta * (energyB - energy))) + (1 - condition) * (torch.exp(beta * (energy - energyB)))
        return torch.cat([observations, x_t], dim=-1), t + time_offset, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.noise_interpolation:
        t = torch.rand(len(actions), 1, device=actions.device)
        x_1 = actions
        x_0 = torch.randn_like(x_1, device=actions.device)
        x_t = (1 - t) * x_0 + t * x_1
        x_0_energy = energy_model.get_scaled_q(obs=observations, act=x_0, scale=argus.energy_scale)
        condition = (energy < x_0_energy).float()
        # weights = condition * (torch.exp(beta * x_0_energy) - torch.exp(beta * energy)) + (1 - condition) * (torch.exp(beta * energy) - torch.exp(beta * x_0_energy))
        # weights = condition * (beta * x_0_energy - beta * energy) + (1 - condition) * (beta * energy - beta * x_0_energy)
        weights = condition * (torch.exp(beta * (x_0_energy - energy))) + (1 - condition) * (torch.exp(beta * (energy - x_0_energy)))
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

    raise NotImplementedError
