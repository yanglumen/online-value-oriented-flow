import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta,Normal

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

class DistFlowMatchingNet(nn.Module):
    def __init__(self, argus, input_dim, output_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.argus = argus
        self.log_prob_clip_range = [-5, 10]
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim), nn.Linear(time_dim, time_dim))
        self.embed_net = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            # nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.last = nn.Sequential(
            nn.Linear(hidden_dim+time_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64) ,
        )
        self.mu_head = nn.Sequential(
            nn.Linear(in_features=64, out_features=output_dim),
            nn.Tanh(),
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(in_features=64, out_features=output_dim),
            nn.Sigmoid(),
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
        xt_mu = self.mu_head(xt) * self.argus.x_t_clip_value
        xt_mu = torch.clamp(xt_mu, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        xt_log_sigma = self.log_sigma_head(xt) * 5
        # xt_log_sigma = torch.clamp(xt_log_sigma, 0, 5)
        dist = Normal(xt_mu, xt_log_sigma.exp())
        return dist, xt_mu, xt_log_sigma

    def sample_action(self, x, t):
        reshape_size = False
        data_num = len(x)
        if len(x.shape) == 3:
            reshape_size = True
            x = torch.reshape(x, (-1, x.shape[-1]))
            t = torch.reshape(t, (-1, t.shape[-1]))
        current_dists, xt_mu, xt_log_sigma = self.forward(x, t)
        sampled_a = current_dists.sample().clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        if self.argus.grpo_exploration_rate is not None:
            sampled_a = torch.randn_like(sampled_a, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value) * self.argus.grpo_exploration_rate + sampled_a
        log_probs = current_dists.log_prob(sampled_a).clamp(*self.log_prob_clip_range).sum(-1, keepdim=True)
        if reshape_size:
            sampled_a = torch.reshape(sampled_a, [data_num, self.argus.grpo_group_size, self.argus.action_dim])
            xt_mu = torch.reshape(xt_mu, [data_num, self.argus.grpo_group_size, self.argus.action_dim])
            log_probs = torch.reshape(log_probs, [data_num, self.argus.grpo_group_size, 1])
        return sampled_a, xt_mu, log_probs

    def sample_action_and_log_prob(self, x, t, n=1):
        current_dists, xt_mu, xt_log_sigma = self.forward(x, t)
        sampled_a = current_dists.sample(sample_shape=[n])
        log_probs = current_dists.log_prob(sampled_a).clamp(*self.log_prob_clip_range).sum(-1, keepdim=True)
        log_probs = log_probs.permute(1, 0, 2)
        return xt_mu, sampled_a.permute(1, 0, 2), log_probs.squeeze()

    def mean_var_divergence(self, x, t, a):
        current_dists, xt_mu, xt_log_sigma = self.forward(x=x, t=t)
        log_probs = current_dists.log_prob(a).clamp(*self.log_prob_clip_range).sum(-1, keepdim=True)
        return xt_mu, xt_log_sigma.exp(), -log_probs

    def mean_log_var_divergence(self, x, t, a):
        current_dists, xt_mu, xt_log_sigma = self.forward(x=x, t=t)
        log_probs = current_dists.log_prob(a).clamp(*self.log_prob_clip_range).sum(-1, keepdim=True)
        return xt_mu, xt_log_sigma, -log_probs

    def log_prob_mu_entropy(self, x, t, a):
        reshape_size = False
        data_num = len(x)
        if len(x.shape) == 3:
            reshape_size = True
            x = torch.reshape(x, (-1, x.shape[-1]))
            t = torch.reshape(t, (-1, t.shape[-1]))
            a = torch.reshape(a, (-1, a.shape[-1]))
        current_dists, xt_mu, xt_log_sigma = self.forward(x=x, t=t)
        log_probs = current_dists.log_prob(a).clamp(*self.log_prob_clip_range).sum(-1, keepdim=True)
        entropy = current_dists.entropy().sum(-1, keepdim=True)
        if reshape_size:
            xt_mu = torch.reshape(xt_mu, [data_num, self.argus.grpo_group_size, self.argus.action_dim])
            log_probs = torch.reshape(log_probs, [data_num, self.argus.grpo_group_size, 1])
            entropy = torch.reshape(entropy, [data_num, self.argus.grpo_group_size, 1])
        return log_probs, xt_mu, entropy

    def log_prob_and_mu(self, x, t, a):
        reshape_size = False
        data_num = len(x)
        if len(x.shape) == 3:
            reshape_size = True
            x = torch.reshape(x, (-1, x.shape[-1]))
            t = torch.reshape(t, (-1, t.shape[-1]))
            a = torch.reshape(a, (-1, a.shape[-1]))
        current_dists, xt_mu, xt_log_sigma = self.forward(x=x, t=t)
        log_probs = current_dists.log_prob(a).clamp(*self.log_prob_clip_range).sum(-1, keepdim=True)
        if reshape_size:
            xt_mu = torch.reshape(xt_mu, [data_num, self.argus.grpo_group_size, self.argus.action_dim])
            log_probs = torch.reshape(log_probs, [data_num, self.argus.grpo_group_size, 1])
        return log_probs, xt_mu

    def log_prob(self, x, t, a):
        current_dists, xt_mu, xt_log_sigma = self.forward(x=x, t=t)
        log_probs = current_dists.log_prob(a).clamp(*self.log_prob_clip_range).sum(-1, keepdim=True)
        return log_probs

    def inference_action(self, x, t, deterministic=True):
        action_dist, xt_mu, xt_log_sigma = self.forward(x=x, t=t)
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        return action

    def gen_action(self, states, critic, time_start=0.0, time_end=1.0, steps=50,
                   x_t_clip_value=None, multi_mode=False, step_anneal=False, anneal_rate=0.98, num_candidate=8, **kwargs):
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
                v = self.inference_action(torch.cat([states, x], dim=-1), t_tensor)
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
            dt = 1.0 / steps
            # for current_generation_time in range(self.argus.multi_stage_genration):
            step_index = 0
            for t in np.linspace(time_start, time_end, steps):  # 逐步演化
                # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                v = self.inference_action(torch.cat([states, x], dim=-1), t_tensor) #* self.energy_model(x) * scale
                if step_anneal:
                    dt *= anneal_rate
                x = x + v * dt * self.argus.flow_step_scale
                if x_t_clip_value is not None:
                    x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
        return x

