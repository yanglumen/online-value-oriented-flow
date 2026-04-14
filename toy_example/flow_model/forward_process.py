import torch
from toy_example.config.hyperparameter import WeightedSamplesType

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
        data, energy, beta, energy_model, weighted_samples_type, time_offset=0, norm_direction=False):
    if weighted_samples_type == WeightedSamplesType.subopt2opt:
        perm = torch.randperm(len(data))
        dataB = data[perm]
        energyB = energy[perm]
        t = torch.rand(len(data), 1, device=data.device)
        condition = (energy < energyB).float()
        x_t = condition * (data * (1 - t) + dataB * t) + (1 - condition) * (data * t + dataB * (1 - t))
        dx_dt = condition * (dataB - data) + (1 - condition) * (data - dataB)
        weights = condition * (torch.exp(beta * energyB) - torch.exp(beta * energy)) + (1 - condition) * (torch.exp(beta * energy) - torch.exp(beta * energyB))
        # weights = condition * (torch.exp(beta * (energyB - energy))) + (1 - condition) * (torch.exp(beta * (energy - energyB)))
        return x_t, t+time_offset, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.noise_interpolation:
        t = torch.rand(len(data), 1, device=data.device)
        x_1 = data
        x_0 = torch.randn_like(x_1, device=data.device)
        x_t = (1 - t) * x_0 + t * x_1
        x_0_energy = energy_model(x_0)
        condition = (energy < x_0_energy).float()
        weights = condition * (torch.exp(beta * x_0_energy) - torch.exp(beta * energy)) + (1 - condition) * (torch.exp(beta * energy) - torch.exp(beta * x_0_energy))
        # weights = condition * (torch.exp(beta * (x_0_energy - energy))) + (1 - condition) * (torch.exp(beta * (energy - x_0_energy)))
        dx_dt = condition * (x_0 - x_1) + (1 - condition) * (x_1 - x_0)
        return x_t, t, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.linear_interpolation:
        t = torch.rand(len(data), 1, device=data.device)
        x_1 = data
        x_0 = torch.randn_like(x_1, device=data.device)
        x_t = (1 - t) * x_0 + t * x_1
        dx_dt = x_1 - x_0
        if norm_direction:
            dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        weights = beta*torch.ones_like(t, device=data.device)
        return x_t, t, dx_dt, weights
    if weighted_samples_type == WeightedSamplesType.random_direction:
        t = torch.rand(len(data), 1, device=data.device)
        x_1 = data
        x_0 = torch.randn_like(x_1, device=data.device)
        x_t = (1 - t) * x_0 + t * x_1
        dx_dt = torch.randn_like(x_1, device=data.device)
        if norm_direction:
            dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        weights = beta*torch.ones_like(t, device=data.device)
        return x_t, t, dx_dt, weights

