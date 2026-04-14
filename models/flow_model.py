import numpy as np
import torch
import torch.nn as nn
from config.multistep_rl_flow_hyperparameter import *
import copy

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
    self.register_buffer("W", torch.randn(embed_dim // 2) * scale)
    # self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FlowMatchingNet(nn.Module):
    def __init__(self, argus, input_dim, output_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.argus = argus
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
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
            nn.Linear(64, output_dim) ,
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

    def gen_action_lagecy(self, states, critic, executed_actions, time_start=0.0, time_end=1.0, steps=50,
                   x_t_clip_value=None, step_anneal=False, anneal_rate=0.98, num_candidate=16, **kwargs):
        if self.argus.multi_mode_action_evaluation:
            num_samples = len(states)
            multi_states = states.unsqueeze(dim=1).repeat(1, num_candidate, 1)
            x = torch.randn(num_samples, num_candidate, self.argus.action_dim, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            dt = 1.0 / steps
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, num_candidate, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([multi_states, x], dim=-1), t_tensor)
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
            x = x.clamp(-self.argus.max_action_val, self.argus.max_action_val)
            if self.argus.rl_mode in [RLTrainMode.flow_constrained_rl4]:
                adv = critic.get_adv(observations=multi_states, actions=x,  last_actions=executed_actions, x_t=None, dx_dt=None, t=None)
                candidate_index = torch.argmax(adv, dim=1, keepdim=True)
                x = torch.gather(x, dim=1, index=candidate_index.expand(-1, -1, self.argus.action_dim))
                x = x.squeeze(dim=1)
            else:
                raise NotImplementedError
        else:
            num_samples = len(states)
            x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            dt = 1.0 / steps
            # for current_generation_time in range(self.argus.multi_stage_genration):
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):  # 逐步演化
                # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([states, x], dim=-1), t_tensor) #* self.energy_model(x) * scale
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
        return x

    def _init_action_latent(self, states, x_t_clip_value=None, deterministic=False):
        num_samples = len(states)
        if deterministic:
            x = torch.zeros(num_samples, self.argus.action_dim, device=self.argus.device)
        else:
            x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device)
        if x_t_clip_value is not None:
            x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
        return x

    def gen_action(self, states, critic, executed_actions, time_start=0.0, time_end=1.0, steps=50,
                   x_t_clip_value=None, step_anneal=False, anneal_rate=0.98, num_candidate=16,
                   deterministic=False, **kwargs):
        if self.argus.multi_mode_action_evaluation:
            value_record, intermediate_action_record = [], []
            num_samples = len(states)
            x = self._init_action_latent(states=states, x_t_clip_value=x_t_clip_value, deterministic=deterministic)
            dt = 1.0 / steps
            # for current_generation_time in range(self.argus.multi_stage_genration):
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):  # 逐步演化
                # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([states, x], dim=-1), t_tensor)  # * self.energy_model(x) * scale
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                adv = critic.get_adv(observations=states, x_t=x, dx_dt=v, t=t_tensor)
                value_record.append(adv)
                intermediate_action_record.append(x.unsqueeze(dim=1))
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
            value_record = torch.cat(value_record, dim=-1)
            intermediate_action_record = torch.cat(intermediate_action_record, dim=1)
            idx = value_record.argmax(dim=1)
            idx_expanded = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, intermediate_action_record.size(2))
            x = torch.gather(intermediate_action_record, dim=1, index=idx_expanded).squeeze(1)
        else:
            num_samples = len(states)
            x = self._init_action_latent(states=states, x_t_clip_value=x_t_clip_value, deterministic=deterministic)
            dt = 1.0 / steps
            # for current_generation_time in range(self.argus.multi_stage_genration):
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):  # 逐步演化
                # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([states, x], dim=-1), t_tensor) #* self.energy_model(x) * scale
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
        return x

    def gen_action_and_Q_values(self, states, critic, executed_actions, time_start=0.0, time_end=1.0, steps=50,
                   x_t_clip_value=None, step_anneal=False, anneal_rate=0.98, num_candidate=16, **kwargs):
        value_record = []
        num_samples = len(states)
        x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device)
        if x_t_clip_value is not None:
            x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
        dt = 1.0 / steps
        # for current_generation_time in range(self.argus.multi_stage_genration):
        step_index = 0
        for t in np.linspace(time_start, time_end, steps):  # 逐步演化
            # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
            t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
            v = self.forward(torch.cat([states, x], dim=-1), t_tensor)  # * self.energy_model(x) * scale
            if step_anneal:
                dt *= anneal_rate
            x = x + v * dt * self.argus.flow_step_scale
            adv = critic.get_adv(observations=states, x_t=x, dx_dt=v, t=t_tensor)
            value_record.append(adv)
            if x_t_clip_value is not None:
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            step_index += 1
        return x, value_record

    def behavior_action(self, states, time_start=0.0, time_end=1.0, steps=50, x_t_clip_value=None, step_anneal=False,
                        anneal_rate=0.98, deterministic=False, **kwargs):
        with torch.no_grad():
            states_shape = [_ for _ in states.shape]
            states_shape[-1] = self.argus.action_dim
            if deterministic:
                x = torch.zeros(*states_shape, device=self.argus.device)
            else:
                x = torch.randn(*states_shape, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            dt = 1.0 / steps
            # for current_generation_time in range(self.argus.multi_stage_genration):
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):  # 逐步演化
                # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
                t_tensor = torch.full((states_shape[0], 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([states, x], dim=-1), t_tensor)  # * self.energy_model(x) * scale
                if step_anneal:
                    dt *= anneal_rate
                # x = x + v * dt * self.argus.flow_step_scale
                x = x + v * dt
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
        return x

class FlowStateNet(nn.Module):
    def __init__(self, argus, input_dim, output_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.argus = argus
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
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
            nn.Linear(64, output_dim) ,
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

    def gen_action_lagecy(self, states, critic, executed_actions, time_start=0.0, time_end=1.0, steps=50,
                   x_t_clip_value=None, step_anneal=False, anneal_rate=0.98, num_candidate=16, **kwargs):
        if self.argus.multi_mode_action_evaluation:
            num_samples = len(states)
            multi_states = states.unsqueeze(dim=1).repeat(1, num_candidate, 1)
            x = torch.randn(num_samples, num_candidate, self.argus.action_dim, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            dt = 1.0 / steps
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, num_candidate, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([multi_states, x], dim=-1), t_tensor)
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
            x = x.clamp(-self.argus.max_action_val, self.argus.max_action_val)
            if self.argus.rl_mode in [RLTrainMode.flow_constrained_rl4]:
                adv = critic.get_adv(observations=multi_states, actions=x,  last_actions=executed_actions, x_t=None, dx_dt=None, t=None)
                candidate_index = torch.argmax(adv, dim=1, keepdim=True)
                x = torch.gather(x, dim=1, index=candidate_index.expand(-1, -1, self.argus.action_dim))
                x = x.squeeze(dim=1)
            else:
                raise NotImplementedError
        else:
            num_samples = len(states)
            x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            dt = 1.0 / steps
            # for current_generation_time in range(self.argus.multi_stage_genration):
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):  # 逐步演化
                # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([states, x], dim=-1), t_tensor) #* self.energy_model(x) * scale
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
        return x

    def gen_action(self, states, inverse_dynamics, critic, executed_actions, time_start=0.0, time_end=1.0, steps=50,
                   x_t_clip_value=None, step_anneal=False, anneal_rate=0.98, num_candidate=16, **kwargs):
        num_samples = len(states)
        x = torch.randn(num_samples, self.argus.observation_dim, device=self.argus.device)
        if x_t_clip_value is not None:
            x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
        dt = 1.0 / steps
        # for current_generation_time in range(self.argus.multi_stage_genration):
        step_index = 0
        for t in np.linspace(time_start, time_end, steps):  # 逐步演化
            # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
            t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
            v = self.forward(torch.cat([states, x], dim=-1), t_tensor) #* self.energy_model(x) * scale
            if step_anneal:
                dt *= anneal_rate
            x = x + v * dt * self.argus.flow_step_scale
            if x_t_clip_value is not None:
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            step_index += 1
        x = inverse_dynamics(torch.cat([states, x], dim=-1))
        return x

class FlowValueNet(nn.Module):
    def __init__(self, argus, input_dim, output_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.argus = argus
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
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
            nn.Linear(64, output_dim) ,
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

    def pred_value(self, states, time_start=0.0, time_end=1.0, steps=10, x_t_clip_value=None, step_anneal=False,
                   anneal_rate=0.98, num_candidate=16, **kwargs):
        num_samples = len(states)
        x = torch.randn(num_samples, 1, device=self.argus.device)
        if x_t_clip_value is not None:
            x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
        dt = 1.0 / steps
        # for current_generation_time in range(self.argus.multi_stage_genration):
        step_index = 0
        for t in np.linspace(time_start, time_end, steps):  # 逐步演化
            # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
            t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
            v = self.forward(torch.cat([states, x], dim=-1), t_tensor)  # * self.energy_model(x) * scale
            if step_anneal:
                dt *= anneal_rate
            x = x + v * dt
            if x_t_clip_value is not None:
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            step_index += 1
        return x

class LargeFlowMatchingNet(nn.Module):
    def __init__(self, argus, input_dim, output_dim, hidden_dim=512, time_dim=64):
        super().__init__()
        self.argus = argus
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
        self.embed_net_1 = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
        )
        self.embed_net_2 = nn.Sequential(
            nn.Linear(hidden_dim*2 + time_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.last = nn.Sequential(
            nn.Linear(hidden_dim+time_dim, 128),
            SiLU(),
            nn.Linear(128, output_dim) ,
        )

    def forward(self, x, t, eps=1e-3):
        t = t * (1. - eps) + eps
        time_embed = self.time_embed(t.squeeze())
        if len(time_embed.shape) != len(x.shape):
            time_embed = time_embed.unsqueeze(1).repeat(1, x.shape[1], 1)
        xt = torch.cat([x, time_embed], dim=-1)
        xt = self.embed_net_1(xt)
        xt = torch.cat([xt, time_embed], dim=-1)
        xt = self.embed_net_2(xt)
        xt = torch.cat([xt, time_embed], dim=-1)
        xt = self.last(xt)
        return xt

    def gen_action(self, states, critic, executed_actions, time_start=0.0, time_end=1.0, steps=50,
                   x_t_clip_value=None, step_anneal=False, anneal_rate=0.98, num_candidate=8, **kwargs):
        if self.argus.multi_mode_action_evaluation:
            num_samples = len(states)
            multi_states = states.unsqueeze(dim=1).repeat(1, num_candidate, 1)
            x = torch.randn(num_samples, num_candidate, self.argus.action_dim, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            dt = 1.0 / steps
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, num_candidate, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([multi_states, x], dim=-1), t_tensor)
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
            x = x.clamp(-self.argus.max_action_val, self.argus.max_action_val)
            if self.argus.rl_mode in [RLTrainMode.flow_constrained_rl4]:
                adv = critic.get_adv(observations=multi_states, last_actions=executed_actions, x_t=x, dx_dt=None, t=None)
                candidate_index = torch.argmax(adv, dim=1, keepdim=True)
                x = torch.gather(x, dim=1, index=candidate_index.expand(-1, -1, self.argus.action_dim))
                x = x.squeeze(dim=1)
            else:
                raise NotImplementedError
        else:
            num_samples = len(states)
            x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            dt = 1.0 / steps
            # for current_generation_time in range(self.argus.multi_stage_genration):
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):  # 逐步演化
                # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([states, x], dim=-1), t_tensor) #* self.energy_model(x) * scale
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
        return x

class OneStepFlowMatchingNet(nn.Module):
    def __init__(self, argus, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.argus = argus
        self.embed_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.Linear(input_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 预测向量场 v(x, t)
        )
        self.last = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            SiLU(),
            nn.Linear(64, output_dim) ,
        )

    def forward(self, x, eps=1e-3):
        xt = self.embed_net(x)
        xt = self.last(xt)
        return xt

    def gen_action(self, states, critic, time_start=0.0, time_end=1.0, steps=50,
                   x_t_clip_value=None, multi_mode=False, step_anneal=False, anneal_rate=0.98, num_candidate=8):
        if multi_mode:
            num_samples = len(states)
            multi_states = states.unsqueeze(dim=1).repeat(1, num_candidate, 1)
            x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            states = states.unsqueeze(dim=1)
            x = x.unsqueeze(dim=1)
            dt = 1.0 / steps
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, 1, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.forward(torch.cat([states, x], dim=-1), t_tensor)
                noise_v = torch.rand(size=(num_samples, num_candidate-1, 1), device=self.argus.device).clamp(min=-1.0, max=1.0) * 0.1
                candidate_v = torch.cat([v, noise_v+v], dim=1)
                multi_x = x.repeat(1, num_candidate, 1)
                adv = critic.get_adv(observations=multi_states, x_t=multi_x, dx_dt=candidate_v, t=t_tensor)
                candidate_index = torch.argmax(adv, dim=1, keepdim=True)
                v = torch.gather(candidate_v, dim=1, index=candidate_index.expand(-1, -1, self.argus.action_dim))
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
            x = x.squeeze(dim=1)
        else:
            num_samples = len(states)
            x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device)
            if x_t_clip_value is not None:
                x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
            x = self.forward(torch.cat([states, x], dim=-1))
        return x


class InverseDynamicsFlowMatchingNet(object):
    def __init__(self, argus, flow, inverse_dynamic):
        super().__init__()
        self.argus = argus
        self.flow = flow
        self.inverse_dynamic = inverse_dynamic

    def gen_action(self, states, time_start=0.0, time_end=1.0, steps=50, x_record=False, enneal=False):
        num_samples = len(states)
        x = torch.randn(num_samples, self.argus.observation_dim, device=self.argus.device)
        dt = 1.0 / steps
        multi_x = []
        # for current_generation_time in range(self.argus.multi_stage_genration):
        step_index = 0
        for t in np.linspace(time_start, time_end, steps):  # 逐步演化
            # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
            t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
            v = self.flow(torch.cat([states, x], dim=-1), t_tensor) #* self.energy_model(x) * scale
            x = x + v * dt * self.argus.flow_step_scale
            step_index += 1
            if x_record:
                if step_index == 40:
                    multi_x.append(x)
        x = self.inverse_dynamic(torch.cat([states, x], dim=-1))
        return x
