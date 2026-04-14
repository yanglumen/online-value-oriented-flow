import os.path
import time

import wandb
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.rl_flow_forward_process import sample_weighted_interpolated_points
from trainer.trainer_util import (
    batch_to_device,
)
from termcolor import colored
from config.dict2class import obj2dict
from config.multistep_rl_flow_hyperparameter import *
from evaluation.sequential_eval import parallel_d4rl_eval_score_function_version, visualization_d4rl

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

def cycle_dataloader(argus, dataset, train_batch_size):
    # random.seed(argus.seed)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True)
    while True:
        for data in dataset_loader:
            yield data
        print("Finish this epoch dataloader !!!!!!!")
        # random.seed(np.random.randint(0, 9999))
        # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True)
        # random.seed(argus.seed)

# class self_Setting(Settings):
#     def __init__(self):
#         super().__init__()
#         self.new_set()
#
#     def new_set(self):
#         self.requirements_collect = False


class guided_flow_trainer():
    def get_detailed_save_path(self):
        return os.path.join(
            self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.current_exp_label)

    def __init__(
            self, argus, train_flow, target_train_flow,
            behavior_flow, flow_energy_model, energy_model, dataset):
        self.argus = argus
        self.model = train_flow
        self.model.to(self.argus.device)
        self.target_model = target_train_flow
        self.target_model.to(self.argus.device)
        self.behavior_flow = behavior_flow
        self.behavior_flow.to(self.argus.device)
        self.flow_energy_model = flow_energy_model
        self.flow_energy_model.to(self.argus.device)
        # self.flow_value_func = flow_value_func
        # self.flow_value_func.to(self.argus.device)
        # self.target_flow_value_func = target_flow_value_func
        # self.target_flow_value_func.to(self.argus.device)
        # self.flow_v_value = flow_v_value
        # self.flow_v_value.to(self.argus.device)
        self.energy_model = energy_model
        self.energy_model.to(self.argus.device)
        self.dataset = dataset
        self.device = argus.device
        self.loss_fn = nn.MSELoss()
        self.ema = EMA(argus.ema_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=argus.lr)
        self.bf_optimizer = torch.optim.Adam(self.behavior_flow.parameters(), lr=argus.lr)
        if self.argus.rl_mode in [RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl3, RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
            self.q_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
            self.fv_v_optimizer = torch.optim.Adam(self.flow_energy_model.flow_v.parameters(), lr=argus.lr)
        elif self.argus.rl_mode in [RLTrainMode.flow_constrained_rl, RLTrainMode.direct_flow_2_result]:
            self.q_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
        else:
            self.flow_value_func = self.flow_energy_model.v
            self.fv_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.fv_v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
        self.energy_q_optimizer = torch.optim.Adam(self.energy_model.q.parameters(), lr=argus.lr)
        if argus.critic_type in [CriticType.iql, CriticType.ciql, CriticType.cql]:
            self.energy_v_optimizer = torch.optim.Adam(self.energy_model.v.parameters(), lr=argus.lr)
        elif argus.critic_type in [CriticType.isql]:
            pass
        else:
            raise NotImplementedError
        self.step = 0
        self.save_path = argus.save_path
        self.save_freq = argus. save_freq
        self.env_name = argus.dataset
        self.wandb_log = argus.wandb_log
        self.wandb_exp_name = argus.wandb_exp_name
        self.wandb_exp_group = argus.wandb_exp_group
        self.wandb_log_frequency = argus.wandb_log_frequency
        self.wandb_project_name = argus.wandb_project_name
        self.best_model_info = {"guidance_scale": 0.0, "performance": 0.0, "performance_std": 0.0}
        self.dataloader = cycle_dataloader(argus=self.argus, dataset=self.dataset, train_batch_size=argus.batch_size)
        if self.wandb_log:
            wandb.init(name=self.wandb_exp_name, group=self.wandb_exp_group, project=self.wandb_project_name, config=obj2dict(self.argus))

    # def expectile_flow_train(self, observations, actions):
    #     s_x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
    #         argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
    #         energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
    #     )
    #     # pred_dx_dt = self.model(s_x_t, t)
    #     # normed_pred_dx_dt = pred_dx_dt / torch.norm(pred_dx_dt, dim=-1, keepdim=True)
    #     # flow_loss = -self.expectile_q(torch.cat([s_x_t, normed_pred_dx_dt], dim=1))
    #
    #     with torch.no_grad():
    #         normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
    #         eq_value = self.expectile_q(x=torch.cat([s_x_t, normed_dx_dt], dim=1), t=t)
    #         ev_value = self.expectile_v(x=s_x_t, t=t)
    #         u = eq_value - ev_value
    #         weights = torch.where(u > 0, self.argus.expectile_flow_tau, (1 - self.argus.expectile_flow_tau)).detach()
    #     pred_dx_dt = self.model(s_x_t, t)
    #     flow_loss = weights * torch.norm(pred_dx_dt - dx_dt.detach(), dim=-1, keepdim=True)
    #     flow_loss = flow_loss.mean()
    #     self.optimizer.zero_grad()
    #     flow_loss.backward()
    #     self.optimizer.step()
    #     self.eq_optimizer.zero_grad()
    #     return {"flow_loss": flow_loss.detach().cpu().numpy().item()}
    #
    # def expectile_rl_flow_train(self, observations, actions):
    #     s_x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
    #         argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
    #         energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
    #     )
    #     pred_dx_dt = self.model(s_x_t, t)
    #     # todo corresponding to the strategy that uses normed direction dxdt to train rl-based flow
    #     normed_pred_dx_dt = pred_dx_dt / torch.norm(pred_dx_dt, dim=-1, keepdim=True)
    #     flow_loss = -self.expectile_q(x=torch.cat([s_x_t, normed_pred_dx_dt], dim=1), t=t)
    #     # todo ciql_expectile_rl_time_advantage
    #     # normed_pred_dx_dt = pred_dx_dt / torch.norm(pred_dx_dt, dim=-1, keepdim=True)
    #     # adv = self.expectile_q(x=torch.cat([s_x_t, normed_pred_dx_dt], dim=1), t=t) - self.expectile_v(x=s_x_t, t=t)
    #     # flow_loss = -adv
    #     # todo corresponding to the strategy that directly uses direction dxdt to train rl-based flow
    #     # flow_loss = -self.expectile_q(x=torch.cat([s_x_t, pred_dx_dt], dim=1), t=t)
    #     flow_loss = flow_loss.mean()
    #     self.optimizer.zero_grad()
    #     flow_loss.backward()
    #     self.optimizer.step()
    #     self.eq_optimizer.zero_grad()
    #     return {"flow_loss": flow_loss.detach().cpu().numpy().item()}

    def flow_direction_divergence(self, observations, x_t, t, need_grad, mode='cosine_divergence'):
        if mode == 'cosine_divergence':
            if need_grad:
                pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
                with torch.no_grad():
                    behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
                cos_sim = F.cosine_similarity(pred_u, behavior_u.detach(), dim=-1)
            else:
                with torch.no_grad():
                    pred_u = self.target_model(x=torch.cat([observations, x_t], dim=-1), t=t)
                    behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
                cos_sim = F.cosine_similarity(pred_u, behavior_u, dim=-1).detach()
            return 1 - cos_sim.unsqueeze(dim=-1)
        raise NotImplementedError

    def flow_value_train_discrete_flow_time(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        with torch.no_grad():
            pred_u = self.target_model(x=torch.cat([observations, x_t], dim=-1), t=t)
        dt = 1 / self.argus.flow_step
        next_t = (dt + t).clamp(min=0, max=1.0)
        flag = (next_t >= 0.9999).float()
        x_t_plus_deltat = x_t + pred_u * (next_t - t)
        fv = self.flow_value_func(x=torch.cat([observations, x_t], dim=-1), t=t)
        with torch.no_grad():
            divergence = self.flow_direction_divergence(observations, x_t, t, need_grad=False)
            target_fv = self.target_flow_value_func(x=torch.cat([observations, x_t_plus_deltat], dim=-1), t=t)
            # target_fv = self.energy_model.get_scaled_q(obs=observations, act=x_t_plus_deltat, scale=self.argus.energy_scale)
            terminal_target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
            target = target_fv * (1 - flag) + terminal_target_fv * flag
        noise_action = x_t + torch.randn_like(x_t)
        consevative_loss = torch.logsumexp(self.flow_value_func(x=torch.cat([observations, noise_action], dim=-1), t=t), dim=0)

        loss = self.loss_fn(fv, -divergence.detach() + self.argus.divergence_discount * target.detach()) + consevative_loss.mean()
        self.fv_optimizer.zero_grad()
        loss.backward()
        self.fv_optimizer.step()
        self.ema.update_model_average(self.target_flow_value_func, self.flow_value_func)
        return {
            "value_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_value_loss": loss.detach().cpu().numpy().item(),
        }

    def flow_value_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value
        )
        with torch.no_grad():
            pred_u = self.target_model(x=torch.cat([observations, x_t], dim=-1), t=t)
        dt = 1/self.argus.flow_step
        next_t = (dt + t).clamp(min=0, max=1.0)
        flag = (next_t >= 1.0).float()
        x_t_plus_deltat = x_t + pred_u * dt #(next_t - t)
        x_t_plus_deltat = torch.clip(x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)

        # normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        # fv = self.flow_value_func(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
        # with torch.no_grad():
        #     divergence = self.flow_direction_divergence(observations, x_t, t, need_grad=False)
        #     target_fv = self.energy_model.get_scaled_q(obs=observations, act=x_t_plus_deltat, scale=self.argus.energy_scale)
        #     terminal_target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        #     target = target_fv + terminal_target_fv
        #     # divergence = self.flow_direction_divergence(observations, x_t, t, need_grad=False)
        #     # next_dx_dt = self.target_model(x=torch.cat([observations, x_t_plus_deltat], dim=-1), t=t)
        #     # normed_next_dx_dt = next_dx_dt / torch.norm(next_dx_dt, dim=-1, keepdim=True)
        #     # target_fv = self.target_flow_value_func(
        #     #     x=torch.cat([observations, x_t_plus_deltat, normed_next_dx_dt], dim=-1), t=t)
        #     # target_fv = self.energy_model.get_scaled_q(obs=observations, act=x_t_plus_deltat, scale=self.argus.energy_scale)
        #     # terminal_target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        #     # target = target_fv*(1-flag) + terminal_target_fv*flag
        # noise_dx_dt = dx_dt + torch.randn_like(dx_dt)
        # normed_noise_dx_dt = noise_dx_dt / torch.norm(noise_dx_dt, dim=-1, keepdim=True)
        # conservative_loss = torch.logsumexp(self.flow_value_func(
        #     x=torch.cat([observations, x_t, normed_noise_dx_dt], dim=-1), t=t), dim=0)
        # loss = self.loss_fn(fv, self.argus.divergence_discount*target.detach()) + self.argus.conservative_coef * conservative_loss.mean()

        # fv = self.flow_value_func(x=torch.cat([observations, x_t], dim=-1), t=t)
        # with torch.no_grad():
        #     dx_dt_plus_deltat = self.target_model(x=torch.cat([observations, x_t_plus_deltat], dim=-1), t=t)
        #     pred_x_1 = x_t_plus_deltat + dx_dt_plus_deltat * (1.0 - next_t)
        #     divergence = self.flow_direction_divergence(observations, x_t, t, need_grad=False)
        #     target_fv = self.energy_model.get_scaled_q(obs=observations, act=pred_x_1, scale=self.argus.energy_scale)
        #     target = target_fv
        # noise_action = x_t + torch.randn_like(x_t)
        # conservative_loss = torch.logsumexp(self.flow_value_func(x=torch.cat([observations, noise_action], dim=-1), t=t), dim=0)

        fv = self.flow_value_func(x=torch.cat([observations, x_t], dim=-1), t=t)
        with torch.no_grad():
            divergence = self.flow_direction_divergence(observations, x_t, t, need_grad=False)
            # target_fv = self.target_flow_value_func(x=torch.cat([observations, x_t_plus_deltat], dim=-1), t=t)
            target_fv = self.energy_model.get_scaled_q(obs=observations, act=x_t_plus_deltat, scale=self.argus.energy_scale)
            terminal_target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
            target = target_fv*(1-flag) + terminal_target_fv*flag
        # noise_action = x_t + torch.randn_like(x_t)
        # conservative_loss = torch.logsumexp(self.flow_value_func(x=torch.cat([observations, noise_action], dim=-1), t=t), dim=0)
        # bound_loss = torch.where(fv > terminal_target_fv, (fv - terminal_target_fv.detach()) ** 2, torch.zeros_like(fv))

        loss = self.loss_fn(fv, -divergence.detach() + self.argus.divergence_discount*target.detach()) #+ self.argus.conservative_coef * conservative_loss.mean()
        self.fv_optimizer.zero_grad()
        loss.backward()
        self.fv_optimizer.step()
        self.ema.update_model_average(self.target_flow_value_func, self.flow_value_func)
        return {
            "value_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_value_loss": loss.detach().cpu().numpy().item(),
        }

    def flow_policy_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )

        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        # pred_u = torch.randn_like(pred_u, device=pred_u.device).clamp(min=-1.0, max=1.0) * 0.1
        dt = 1 / self.argus.flow_step
        next_t = (dt + t).clamp(min=0, max=1.0)
        x_t_plus_deltat = x_t + pred_u * dt #(next_t - t)
        x_t_plus_deltat = torch.clip(x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        divergence = self.flow_direction_divergence(observations, x_t, t, need_grad=True)

        # normed_pred_u = pred_u / torch.norm(pred_u, dim=-1, keepdim=True)
        # loss = - self.flow_value_func(
        #     x=torch.cat([observations, x_t, normed_pred_u], dim=-1), t=t)

        # loss = self.argus.divergence_coef * divergence - self.flow_value_func(
        #     x=torch.cat([observations, x_t_plus_deltat], dim=-1), t=t)
        loss = - self.flow_value_func(x=torch.cat([observations, x_t_plus_deltat], dim=-1), t=t)

        # pred_x_1 = x_t + pred_u * (1.0 - t)
        # loss = - self.energy_model.get_scaled_q(obs=observations, act=pred_x_1, scale=self.argus.energy_scale)

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def multistep_flow_value_train(self, observations, actions):
        x_t, t_start, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.same_t_step,
            clip_value=self.argus.x_t_clip_value
        )
        num_samples = len(observations)
        dt = 1.0 / self.argus.flow_step
        step_index = 0
        total_divergence = 0
        noise_action = x_t + torch.randn_like(x_t)
        conservative_loss = torch.logsumexp(
            self.flow_value_func(
                x=torch.cat([observations, noise_action], dim=-1),
                t=torch.full((num_samples, 1), t_start, dtype=torch.float32, device=self.argus.device)), dim=0)
        fv = self.flow_value_func(
            x=torch.cat([observations, x_t], dim=-1),
            t=torch.full((num_samples, 1), t_start, dtype=torch.float32, device=self.argus.device))
        with torch.no_grad():
            t = t_start
            while t < 1.0:
                t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
                divergence = self.flow_direction_divergence(observations, x_t, t_tensor, need_grad=False)
                total_divergence += divergence
                pred_u = self.target_model(x=torch.cat([observations, x_t], dim=-1), t=t_tensor)
                x_t = x_t + pred_u * np.minimum(dt, 1-t)
                x_t = torch.clip(x_t, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                t += dt
                step_index += 1
            target_fv = -total_divergence + self.energy_model.get_scaled_q(
                obs=observations, act=actions, scale=self.argus.energy_scale)
        loss = self.loss_fn(fv, target_fv.detach()) + self.argus.conservative_coef * conservative_loss.mean()
        loss = loss.mean()
        self.fv_optimizer.zero_grad()
        loss.backward()
        self.fv_optimizer.step()
        self.ema.update_model_average(self.target_flow_value_func, self.flow_value_func)
        return {
            "value_divergence": total_divergence.mean().detach().cpu().numpy().item(),
            "flow_value_loss": loss.detach().cpu().numpy().item(),
            "flow_value": fv.mean().detach().cpu().numpy().item(),
        }

    def rl_like_flow_value_train(self, observations, actions):
        x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        noise_x_t = (x_t + torch.randn_like(x_t)).clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        noise_dxdt = (dx_dt + torch.randn_like(dx_dt)).clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t)
        conservative_q_loss = torch.logsumexp(
            self.flow_energy_model.q(x=torch.cat([observations, noise_x_t, noise_dxdt], dim=-1), t=t), dim=0) - Q.mean()
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        # normed_noise_dxdt = noise_dxdt / torch.norm(noise_dxdt, dim=-1, keepdim=True)
        V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
        # conservative_v_loss = torch.logsumexp(
        #     self.flow_energy_model.v(x=torch.cat([observations, noise_x_t, normed_dx_dt], dim=-1), t=t), dim=0) - V.mean()
        with torch.no_grad():
            target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        Q_loss = self.loss_fn(Q, target_fv.detach()) + self.argus.conservative_coef * conservative_q_loss.mean()
        V_loss = self.loss_fn(V, target_fv.detach())# + self.argus.conservative_coef * conservative_v_loss.mean()

        # x_t, t_start, dx_dt, weights = sample_weighted_interpolated_points(
        #     argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
        #     energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.same_t_step,
        #     clip_value=self.argus.x_t_clip_value
        # )
        # num_samples = len(observations)
        # t_tensor = torch.full((num_samples, 1), t_start, dtype=torch.float32, device=self.argus.device)
        # dt = 1.0 / self.argus.flow_step
        # step_index = 0
        # noise_x_t = x_t + torch.randn_like(x_t)
        # noise_dxdt = dx_dt + torch.randn_like(dx_dt)
        # conservative_loss = torch.logsumexp(
        #     self.flow_value_func(
        #         x=torch.cat([observations, noise_x_t, noise_dxdt], dim=-1), t=t_tensor), dim=0)
        # # conservative_loss = self.flow_value_func(x=torch.cat([observations, noise_x_t, noise_dxdt], dim=-1), t=t_tensor)
        # fv = self.flow_value_func(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t_tensor)
        # with torch.no_grad():
        #     # distance_penalty = torch.norm(x_t - actions, dim=-1, keepdim=True) * 10
        #     target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        # # with torch.no_grad():
        # #     t = t_start
        # #     while t < 1.0:
        # #         t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
        # #         # divergence = self.flow_direction_divergence(observations, x_t, t_tensor, need_grad=False)
        # #         pred_u = self.target_model(x=torch.cat([observations, x_t], dim=-1), t=t_tensor)
        # #         distance_penalty = torch.norm(x_t-actions, dim=-1, keepdim=True)
        # #         total_divergence += distance_penalty * 5
        # #         x_t = x_t + pred_u * np.minimum(dt, 1-t)
        # #         # x_t = torch.clip(x_t, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # #         t += dt
        # #         step_index += 1
        # #     target_fv = -total_divergence + self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        # loss = self.loss_fn(fv, target_fv.detach())
        # loss = loss.mean() + self.argus.conservative_coef * conservative_loss.mean()
        self.fv_optimizer.zero_grad()
        Q_loss.backward()
        self.fv_optimizer.step()
        self.fv_v_optimizer.zero_grad()
        V_loss.backward()
        self.fv_v_optimizer.step()
        self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        return {
            # "value_divergence": total_divergence.mean().detach().cpu().numpy().item(),
            "flow_value_loss": Q_loss.detach().cpu().numpy().item(),
            "flow_v_value_loss": V_loss.detach().cpu().numpy().item(),
            "flow_value": Q.mean().detach().cpu().numpy().item(),
            "flow_v_value": V.mean().detach().cpu().numpy().item(),
        }

    def rl_like_flow_policy_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        # # target_pred_u = self.target_model(x=torch.cat([observations, x_t], dim=-1), t=t)
        # pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        # with torch.no_grad():
        #     behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        # cos_sim = F.cosine_similarity(pred_u, behavior_u.detach(), dim=-1)
        # divergence = 1 - cos_sim.unsqueeze(dim=-1)
        #
        # with torch.no_grad():
        #     fv_dx_dt = self.target_flow_value_func(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t)
        #     # fv_pred_u = self.target_flow_value_func(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        #     # weight = torch.softmax((fv_dx_dt - fv_pred_u)/self.argus.weight_policy_regression_coef, dim=0)
        #     # weight = torch.softmax(torch.cat([fv_dx_dt, fv_pred_u], dim=-1)/self.argus.weight_policy_regression_coef, dim=-1)
        #     # loss = weight[:, 0:1] * self.loss_fn(pred_u, dx_dt) + weight[:, 1:] * self.loss_fn(pred_u, target_pred_u.detach())
        #     weight = fv_dx_dt - self.energy_model.get_scaled_v(obs=observations, scale=self.argus.energy_scale)
        #     weight = torch.clamp(torch.exp(weight / self.argus.weight_policy_regression_coef), -50, 50)
        # loss = weight.detach() * torch.sum((pred_u-dx_dt)**2, dim=-1, keepdim=True)
        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        pred_u = pred_u.clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        cos_sim = F.cosine_similarity(pred_u, behavior_u.detach(), dim=-1)
        divergence = 1 - cos_sim.unsqueeze(dim=-1)
        normed_pred_u = pred_u / torch.norm(pred_u, dim=-1, keepdim=True)
        pred_Q = self.flow_energy_model.q_target(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        pred_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_pred_u], dim=-1), t=t)
        loss = -(pred_Q - pred_V)

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def flow_train(self, observations, actions):
        loss = {}
        loss_info = self.rl_like_flow_value_train(observations=observations, actions=actions)
        loss.update(loss_info)
        loss_info = self.rl_like_flow_policy_train(observations=observations, actions=actions)
        loss.update(loss_info)
        return loss

    def energy_train(self, iql_tau, observations, actions, next_observations, rewards, dones, next_actions=None, fake_actions=None, fake_next_actions=None):
        q_loss, v_loss, loss_info = self.energy_model.loss(
            tau=iql_tau, behavior_model=self.behavior_flow, observations=observations, actions=actions, next_observations=next_observations,
            next_actions=next_actions, rewards=rewards, dones=dones, fake_actions=fake_actions, fake_next_actions=fake_next_actions)
        if v_loss is not None:
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.energy_model.v.parameters(), max_norm=10.0)
            self.energy_v_optimizer.step()
            self.energy_v_optimizer.zero_grad()
        if q_loss is not None:
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.energy_model.q.parameters(), max_norm=10.0)
            self.energy_q_optimizer.step()
            self.energy_q_optimizer.zero_grad()
        self.energy_model.q_ema()
        return loss_info

    def behavior_flow_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        v_pred = self.behavior_flow(torch.cat([observations, x_t], dim=-1), t)
        loss = self.loss_fn(v_pred, dx_dt)
        self.bf_optimizer.zero_grad()
        loss.backward()
        self.bf_optimizer.step()
        return {"behavior_flow_loss": loss.detach().cpu().numpy().item()}

    def guided_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            # self.load_best_flow_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = 40
        except:
            print(colored("Load critic checkpoint unsuccessful !!!!", "red"))
            epoch_offset = 0
        for epoch in range(epoch_offset, num_epochs):
            if self.argus.debug_mode:
                update_energy = True
                update_behavior = True
                update_flow = True
                update_flow_value = True
                update_flow_policy = True
            else:
                update_energy = True if epoch < self.argus.update_energy_end_epoch else False
                update_behavior = True if epoch < self.argus.update_behavior_end_epoch else False
                update_flow = False if epoch < self.argus.update_flow_start_epoch else True
                # update_flow_value = True if epoch % 2 == 0 else False
                # update_flow_policy = False if epoch % 2 == 0 else True
                update_flow_value = True
                update_flow_policy = True
                # update_flow_value = True if (epoch < 60 and epoch >= 40) else False
                # # update_flow_value = True if epoch >= 40 else False
                # update_flow_policy = True if epoch >= 60 else False
            # if update_flow_policy:
            #     self.argus.divergence_coef = np.maximum(0.1, self.argus.divergence_coef * 0.95)
            for step in range(num_steps_per_epoch):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                observations = batch.observations.reshape(-1, self.argus.observation_dim)
                actions = batch.actions.reshape(-1, self.argus.action_dim)
                next_observations = batch.next_observations.reshape(-1, self.argus.observation_dim)
                rewards = batch.rewards.reshape(-1, 1)
                dones = batch.dones.reshape(-1, 1)
                loss = {}
                if update_energy:
                    energy_loss_info = self.energy_train(
                        iql_tau=self.argus.iql_tau, observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(energy_loss_info)
                if update_behavior:
                    energy_loss_info = self.behavior_flow_train(observations=observations, actions=actions)
                    loss.update(energy_loss_info)
                if update_flow:
                    flow_loss_info = self.flow_train(observations=observations, actions=actions)
                    loss.update(flow_loss_info)

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        wandb_infos = {}
                        wandb_infos.update(loss)
                        # wandb_infos.update({"diffusion_loss": loss})
                        for info_key, info_val in wandb_infos.items():
                            wandb_infos[info_key] = info_val
                        wandb.log(wandb_infos, step=self.step)


                if (self.step+1) % 5000 == 0 and update_flow_policy:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def flow_constrained_value_train(self, observations, actions, next_observations, rewards, dones):
        if self.argus.rl_mode == RLTrainMode.flow_constrained_rl2:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl2_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl3:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl3_loss(
                flow=self.model, behavior_flow=self.behavior_flow, energy_model=self.energy_model,
                observations=observations, actions=actions, next_observations=next_observations, rewards=rewards,
                dones=dones, tau=self.argus.iql_tau, multiple_actions=self.argus.flow_constrained_rl4_multiple_actions
            )
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl4:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl4_loss(
                flow=self.model, behavior_flow=self.behavior_flow, energy_model=self.energy_model,
                observations=observations, actions=actions, next_observations=next_observations, rewards=rewards,
                dones=dones, tau=self.argus.iql_tau, multiple_actions=self.argus.flow_constrained_rl4_multiple_actions,
            )
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl5:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl5_loss(
                flow=self.model, behavior_flow=self.behavior_flow, energy_model=self.energy_model,
                observations=observations, actions=actions, next_observations=next_observations, rewards=rewards,
                dones=dones, tau=self.argus.iql_tau, multiple_actions=self.argus.flow_constrained_rl5_multiple_actions,
            )
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl:
            Q_loss, V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
        else:
            raise NotImplementedError
        if Q_loss is not None:
            self.q_optimizer.zero_grad()
            Q_loss.backward()
            self.q_optimizer.step()
            self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        if V_loss is not None:
            self.v_optimizer.zero_grad()
            V_loss.backward()
            self.v_optimizer.step()
        return flow_loss_info

    def flow_constrained_policy_train(self, observations, actions, next_observations, next_actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=next_observations, actions=next_actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )

        pred_u = self.model(x=torch.cat([next_observations, x_t], dim=-1), t=t)
        dt = 1 / self.argus.flow_step
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([next_observations, x_t], dim=-1), t=t)
        if self.argus.dataset in self.argus.adroit_dataset:
            pred_u = pred_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            divergence = torch.norm(pred_u - behavior_u.detach(), dim=-1, keepdim=True)
        elif self.argus.dataset in self.argus.maze2d_dataset:
            pred_u = pred_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            divergence = torch.norm(pred_u - behavior_u.detach(), dim=-1, keepdim=True)
        else:
            divergence = torch.norm(pred_u - behavior_u.detach(), dim=-1, keepdim=True)
        x_t_plus_deltat = x_t + pred_u * dt  # (next_t - t)
        x_t_plus_deltat = torch.clip(x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        if self.argus.rl_mode == RLTrainMode.flow_constrained_rl2:
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl4:
            if self.argus.dataset == "halfcheetah-medium-expert-v2":
                # adv = (-self.argus.divergence_coef * divergence + self.flow_energy_model.flow_v(
                #     torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))) - self.flow_energy_model.flow_v(torch.cat([x_t, actions, next_observations], dim=-1))
                # loss = -adv
                loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
            elif self.argus.dataset in self.argus.maze2d_dataset:
                loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
            elif self.argus.dataset in self.argus.adroit_dataset:
                loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
            elif self.argus.dataset in self.argus.antmaze_dataset:
                loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
            else:
                # adv = (-self.argus.divergence_coef * divergence + self.flow_energy_model.flow_v(
                #     torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))) - self.flow_energy_model.flow_v(torch.cat([x_t, actions, next_observations], dim=-1))
                # loss = -adv
                loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl5:
            # loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
            adv = (-self.argus.divergence_coef * divergence + self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1)))-self.flow_energy_model.flow_v(torch.cat([x_t, actions, next_observations], dim=-1))
            loss = -adv
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl3:
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, next_observations], dim=-1))
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl:
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.v(torch.cat([x_t_plus_deltat, next_observations], dim=-1))
        else:
            raise NotImplementedError
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }
    def flow_constrained_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = self.argus.epoch_offset
        except:
            print(colored("Load critic checkpoint fail !!!!", "red"))
            epoch_offset = 0
        for epoch in range(epoch_offset, num_epochs):
            if self.argus.debug_mode:
                update_energy = True
                update_behavior = True
                update_flow = True
            else:
                update_energy = True if epoch < self.argus.update_energy_end_epoch else False
                update_behavior = True if epoch < self.argus.update_behavior_end_epoch else False
                update_flow = False if epoch < self.argus.update_flow_start_epoch else True
            for step in range(num_steps_per_epoch):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                observations = batch.observations.reshape(-1, self.argus.observation_dim)
                actions = batch.actions.reshape(-1, self.argus.action_dim)
                next_observations = batch.next_observations.reshape(-1, self.argus.observation_dim)
                rewards = batch.rewards.reshape(-1, 1)
                dones = batch.dones.reshape(-1, 1)
                loss = {}
                if update_energy:
                    energy_loss_info = self.energy_train(
                        iql_tau=self.argus.iql_tau, observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(energy_loss_info)
                if update_behavior:
                    energy_loss_info = self.behavior_flow_train(observations=observations, actions=actions)
                    loss.update(energy_loss_info)
                if update_flow:
                    flow_loss_info = self.flow_constrained_value_train(observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(flow_loss_info)
                    if self.argus.dataset in self.argus.adroit_dataset:
                        policy_repeat  = 3
                    elif self.argus.dataset in self.argus.antmaze_dataset:
                        policy_repeat  = 1
                    else:
                        policy_repeat = 3
                    for _ in range(policy_repeat):
                        flow_loss_info = self.flow_constrained_policy_train(
                            observations=observations[:-1], actions=actions[:-1], next_observations=observations[1:], next_actions=actions[1:])
                    loss.update(flow_loss_info)

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        wandb_infos = {}
                        wandb_infos.update(loss)
                        # wandb_infos.update({"diffusion_loss": loss})
                        for info_key, info_val in wandb_infos.items():
                            wandb_infos[info_key] = info_val
                        wandb.log(wandb_infos, step=self.step)


                if (self.step+1) % 5000 == 0 and update_flow:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def direct_flow2result_value_train(self, observations, actions, next_observations, rewards, dones):
        Q_loss, V_loss, loss_info = self.flow_energy_model.direct_flow2result_loss(
            observations=observations, actions=actions, next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)

        self.q_optimizer.zero_grad()
        Q_loss.backward()
        self.q_optimizer.step()
        self.v_optimizer.zero_grad()
        V_loss.backward()
        self.v_optimizer.step()
        self.flow_energy_model.q_ema()
        return loss_info

    def direct_flow2result_policy_train(self, observations, actions):
        num_samples = len(observations)
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        pred_u = pred_u.clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        behavior_divergence = torch.norm(pred_u - behavior_u, dim=-1, keepdim=True)
        dt = 1.0 / self.argus.flow_step
        x_t = x_t + pred_u * dt
        next_pred_u = self.model(x=torch.cat([observations, x_t.detach()], dim=-1), t=t+dt)
        direction_consistency = torch.norm(pred_u/torch.norm(pred_u, dim=-1, keepdim=True)-next_pred_u/torch.norm(next_pred_u, dim=-1, keepdim=True), dim=-1, keepdim=True)
        x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        time_start, time_end, steps = 0, 1.0, self.argus.direct_flow_step
        for t in np.linspace(time_start, time_end, steps):
            t_tensor = torch.full((num_samples, 1), t, dtype=torch.float32, device=self.argus.device)
            direct_u = self.model(x=torch.cat([observations, x], dim=-1), t=t_tensor)
            x = x + direct_u * 1/self.argus.direct_flow_step
        x = x.clamp(-self.argus.max_action_val, self.argus.max_action_val)
        pred_Q = self.flow_energy_model.get_scaled_q(obs=observations, act=x, scale=1.0)
        with torch.no_grad():
            pred_V = self.flow_energy_model.get_scaled_v(obs=observations, scale=1.0)
        # loss = - (pred_Q - pred_V) + behavior_divergence + direction_consistency
        loss = - pred_Q + behavior_divergence + direction_consistency

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.q_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": behavior_divergence.mean().detach().cpu().numpy().item(),
            "direction_consistency": direction_consistency.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def direct_flow2result_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = 40
        except:
            print(colored("Load critic checkpoint unsuccessful !!!!", "red"))
            epoch_offset = 0
        for epoch in range(epoch_offset, num_epochs):
            if self.argus.debug_mode:
                update_energy = True
                update_flow = True
                update_flow_value = True
                update_flow_policy = True
            else:
                update_energy = True if epoch < 40 else False
                update_flow = False if epoch < 40 else True
                update_flow_value = True
                update_flow_policy = True
            for step in range(num_steps_per_epoch):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                observations = batch.observations.reshape(-1, self.argus.observation_dim)
                actions = batch.actions.reshape(-1, self.argus.action_dim)
                next_observations = batch.next_observations.reshape(-1, self.argus.observation_dim)
                rewards = batch.rewards.reshape(-1, 1)
                dones = batch.dones.reshape(-1, 1)
                loss = {}
                if update_energy:
                    energy_loss_info = self.energy_train(
                        iql_tau=self.argus.iql_tau, observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(energy_loss_info)
                    energy_loss_info = self.behavior_flow_train(observations=observations, actions=actions)
                    loss.update(energy_loss_info)
                if update_flow:
                    flow_loss_info = self.direct_flow2result_value_train(observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(flow_loss_info)
                    flow_loss_info = self.direct_flow2result_policy_train(observations=observations, actions=actions)
                    loss.update(flow_loss_info)

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        wandb_infos = {}
                        wandb_infos.update(loss)
                        # wandb_infos.update({"diffusion_loss": loss})
                        for info_key, info_val in wandb_infos.items():
                            wandb_infos[info_key] = info_val
                        wandb.log(wandb_infos, step=self.step)


                if (self.step+1) % 5000 == 0 and update_flow_policy:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def adv_based_value_train(self, observations, actions, next_observations, next_actions, multiple_actions=20):
        x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        # noise_x_t = (x_t + torch.randn_like(x_t)).clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # noise_dxdt = (dx_dt + torch.randn_like(dx_dt)).clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t)
        # conservative_q_loss = torch.logsumexp(
        #     self.flow_energy_model.q(x=torch.cat([observations, noise_x_t, noise_dxdt], dim=-1), t=t), dim=0) - Q.mean()
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        # normed_noise_dxdt = noise_dxdt / torch.norm(noise_dxdt, dim=-1, keepdim=True)
        V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
        # conservative_v_loss = torch.logsumexp(
        #     self.flow_energy_model.v(x=torch.cat([observations, noise_x_t, normed_dx_dt], dim=-1), t=t), dim=0) - V.mean()
        with torch.no_grad():
            target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
            # target_fv = self.energy_model.get_scaled_min_qs(obs=observations, act=actions, scale=self.argus.energy_scale)
        Q_loss = self.loss_fn(Q, target_fv.detach())  # + self.argus.conservative_coef * conservative_q_loss.mean()
        V_loss = self.loss_fn(V, target_fv.detach())  # + self.argus.conservative_coef * conservative_v_loss.mean()
        self.fv_optimizer.zero_grad()
        Q_loss.backward()
        self.fv_optimizer.step()
        self.fv_v_optimizer.zero_grad()
        V_loss.backward()
        self.fv_v_optimizer.step()
        self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        return {
            # "value_divergence": total_divergence.mean().detach().cpu().numpy().item(),
            "flow_value_loss": Q_loss.detach().cpu().numpy().item(),
            "flow_v_value_loss": V_loss.detach().cpu().numpy().item(),
            "flow_value": Q.mean().detach().cpu().numpy().item(),
            "flow_v_value": V.mean().detach().cpu().numpy().item(),
        }
        # num_samples = len(observations)
        # t_tensor = torch.randn(num_samples, multiple_actions, 1, device=self.argus.device, dtype=torch.float32)
        # x_1 = actions.unsqueeze(dim=1).repeat(1, multiple_actions, 1)
        # x_0 = torch.randn_like(x_1, device=actions.device)
        # x_0 = torch.clip(x_0, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # x_t = (1 - t_tensor) * x_0 + t_tensor * x_1
        # dx_dt = x_1 - x_0
        # Q = self.flow_energy_model.q(x=torch.cat([observations.unsqueeze(dim=1).repeat(1, multiple_actions, 1), x_t, dx_dt], dim=-1), t=t_tensor)
        # with torch.no_grad():
        #     target_Q = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        # Q_loss = self.loss_fn(Q, target_Q.unsqueeze(1).repeat(1, multiple_actions, 1).detach())
        # normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        # V = self.flow_energy_model.v(x=torch.cat([observations.unsqueeze(dim=1).repeat(1, multiple_actions, 1), x_t, normed_dx_dt], dim=-1), t=t_tensor)
        # V_loss = self.loss_fn(V, target_Q.unsqueeze(1).repeat(1, multiple_actions, 1).detach())
        #
        # self.fv_optimizer.zero_grad()
        # Q_loss.backward()
        # self.fv_optimizer.step()
        # self.fv_v_optimizer.zero_grad()
        # V_loss.backward()
        # self.fv_v_optimizer.step()
        # self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        # return {
        #     # "value_divergence": total_divergence.mean().detach().cpu().numpy().item(),
        #     "flow_value_loss": Q_loss.detach().cpu().numpy().item(),
        #     "flow_v_value_loss": V_loss.detach().cpu().numpy().item(),
        #     "flow_value": Q.mean().detach().cpu().numpy().item(),
        #     "flow_v_value": V.mean().detach().cpu().numpy().item(),
        # }

    def adv_based_policy_train(self, observations, actions, next_observations, next_actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        # with torch.no_grad():
        #     behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        pred_u = pred_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # divergence = torch.norm(pred_u - dx_dt, dim=-1, keepdim=True)
        divergence = self.loss_fn(pred_u, dx_dt)
        normed_pred_u = pred_u / torch.norm(pred_u, dim=-1, keepdim=True)
        pred_Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        pred_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_pred_u], dim=-1), t=t)
        # loss = self.argus.divergence_coef * divergence + pred_V - pred_Q
        loss = pred_V.detach() - pred_Q

        loss = self.argus.divergence_coef * divergence + loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }
        # x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
        #     argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
        #     energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        #     clip_value=self.argus.x_t_clip_value,
        # )
        # pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        # pred_u = pred_u.clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # with torch.no_grad():
        #     behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        # divergence = torch.norm(pred_u - behavior_u, dim=-1, keepdim=True)
        # pred_Q = self.flow_energy_model.q_target(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        # with torch.no_grad():
        #     normed_pred_u = pred_u / torch.norm(pred_u, dim=-1, keepdim=True)
        #     pred_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_pred_u], dim=-1), t=t)
        # # loss = divergence - (pred_Q - pred_V.detach())
        # adv = pred_Q - pred_V.detach()
        # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # loss = - adv
        #
        # loss = loss.mean()
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.fv_optimizer.zero_grad()
        # self.ema.update_model_average(self.target_model, self.model)
        # return {
        #     "flow_divergence_coef": self.argus.divergence_coef,
        #     "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
        #     "flow_loss": loss.detach().cpu().numpy().item(),
        # }

    def adv_based_flow_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = 40
        except:
            print(colored("Load critic checkpoint unsuccessful !!!!", "red"))
            epoch_offset = 0
        train_time_record = time.time()
        for epoch in range(epoch_offset, num_epochs):
            if self.argus.debug_mode:
                update_energy = True
                update_behavior = True
                update_flow = True
            else:
                update_energy = True if epoch < self.argus.update_energy_end_epoch else False
                update_behavior = True if epoch < self.argus.update_behavior_end_epoch else False
                update_flow = False if epoch < self.argus.update_flow_start_epoch else True
            for step in range(num_steps_per_epoch):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                observations = batch.observations.reshape(-1, self.argus.observation_dim)
                actions = batch.actions.reshape(-1, self.argus.action_dim)
                next_observations = batch.next_observations.reshape(-1, self.argus.observation_dim)
                rewards = batch.rewards.reshape(-1, 1)
                dones = batch.dones.reshape(-1, 1)
                loss = {}
                if update_energy:
                    energy_loss_info = self.energy_train(
                        iql_tau=self.argus.iql_tau, observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(energy_loss_info)
                if update_behavior:
                    energy_loss_info = self.behavior_flow_train(observations=observations, actions=actions)
                    loss.update(energy_loss_info)
                if update_flow:
                    flow_loss_info = self.adv_based_value_train(
                        observations=observations[:-1], actions=actions[:-1], next_observations=observations[1:], next_actions=actions[1:], multiple_actions=self.argus.adv_rl_multiple_actions)
                    loss.update(flow_loss_info)
                    flow_loss_info = self.adv_based_policy_train(observations=observations[:-1], actions=actions[:-1], next_observations=observations[1:], next_actions=actions[1:])
                    loss.update(flow_loss_info)

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        wandb_infos = {}
                        wandb_infos.update(loss)
                        # wandb_infos.update({"diffusion_loss": loss})
                        for info_key, info_val in wandb_infos.items():
                            wandb_infos[info_key] = info_val
                        wandb.log(wandb_infos, step=self.step)


                if self.step % 10000 == 0 and update_flow:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch}| Time {time.time()-train_time_record} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                    train_time_record = time.time()
                self.step += 1

    def visualization_intermediate_actions(self):
        try:
            self.load_best_flow_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
        except:
            print(colored("Load critic checkpoint unsuccessful !!!!", "red"))
        print(f"Perform evaluation ......")
        eval_results, visualization_with_Q_value = visualization_d4rl(
            argus=self.argus, dataset=self.dataset, model=self.model,
            critic=self.flow_energy_model, guidance_scale=1.0, eval_episodes=10, ddim_sample=True)
        print(eval_results)
        print(f"Evaluation end ......")

    def grpo_value_train(self, observations, actions, next_observations, next_actions, multiple_actions=20):
        x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        # noise_x_t = (x_t + torch.randn_like(x_t)).clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # noise_dxdt = (dx_dt + torch.randn_like(dx_dt)).clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t)
        # conservative_q_loss = torch.logsumexp(
        #     self.flow_energy_model.q(x=torch.cat([observations, noise_x_t, noise_dxdt], dim=-1), t=t), dim=0) - Q.mean()
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        # normed_noise_dxdt = noise_dxdt / torch.norm(noise_dxdt, dim=-1, keepdim=True)
        V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
        # conservative_v_loss = torch.logsumexp(
        #     self.flow_energy_model.v(x=torch.cat([observations, noise_x_t, normed_dx_dt], dim=-1), t=t), dim=0) - V.mean()
        with torch.no_grad():
            target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
            # target_fv = self.energy_model.get_scaled_min_qs(obs=observations, act=actions, scale=self.argus.energy_scale)
        Q_loss = self.loss_fn(Q, target_fv.detach())  # + self.argus.conservative_coef * conservative_q_loss.mean()
        V_loss = self.loss_fn(V, target_fv.detach())  # + self.argus.conservative_coef * conservative_v_loss.mean()
        self.fv_optimizer.zero_grad()
        Q_loss.backward()
        self.fv_optimizer.step()
        self.fv_v_optimizer.zero_grad()
        V_loss.backward()
        self.fv_v_optimizer.step()
        self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        return {
            # "value_divergence": total_divergence.mean().detach().cpu().numpy().item(),
            "flow_value_loss": Q_loss.detach().cpu().numpy().item(),
            "flow_v_value_loss": V_loss.detach().cpu().numpy().item(),
            "flow_value": Q.mean().detach().cpu().numpy().item(),
            "flow_v_value": V.mean().detach().cpu().numpy().item(),
        }

    def grpo_policy_train(self, observations, actions, next_observations, next_actions, expectile=0.5):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
            behavior_u = behavior_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            normed_behavior_u = behavior_u / torch.norm(behavior_u, dim=-1, keepdim=True)
            behavior_u_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_behavior_u], dim=-1), t=t)
            # behavior_u_V = self.flow_energy_model.q_target(x=torch.cat([observations, x_t, behavior_u], dim=-1), t=t)
            target_pred_u = self.target_model(x=torch.cat([observations, x_t], dim=-1), t=t)
            target_pred_u = target_pred_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            normed_target_pred_u = target_pred_u / torch.norm(target_pred_u, dim=-1, keepdim=True)
            target_pred_u_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_target_pred_u], dim=-1), t=t)
            # target_pred_u_V = self.flow_energy_model.q_target(x=torch.cat([observations, x_t, target_pred_u], dim=-1), t=t)
            Q_bar = torch.max(expectile * behavior_u_V + (1 - expectile) * target_pred_u_V, expectile * target_pred_u_V + (1 - expectile) * behavior_u_V)

        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        pred_u = pred_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        divergence = self.loss_fn(pred_u, dx_dt)
        pred_Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        adv = pred_Q - Q_bar.detach()
        adv = adv / torch.std(adv).detach()
        loss = -adv
        # loss = Q_bar.detach() - pred_Q

        # pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        # pred_u = pred_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        # divergence = self.loss_fn(pred_u, dx_dt)
        # pred_Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        # adv = pred_Q - Q_bar.detach()
        # adv = (adv - torch.mean(adv).detach()) / torch.std(adv).detach()
        # loss = -adv
        # # loss = Q_bar.detach() - pred_Q

        loss = self.argus.divergence_coef * divergence + loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def grpo_flow_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = 40
        except:
            print(colored("Load critic checkpoint unsuccessful !!!!", "red"))
            epoch_offset = 0
        print(colored("grpo mode train !!!!", "red"))
        for epoch in range(epoch_offset, num_epochs):
            if self.argus.debug_mode:
                update_energy = True
                update_behavior = True
                update_flow = True
            else:
                update_energy = True if epoch < self.argus.update_energy_end_epoch else False
                update_behavior = True if epoch < self.argus.update_behavior_end_epoch else False
                update_flow = False if epoch < self.argus.update_flow_start_epoch else True
            for step in range(num_steps_per_epoch):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                observations = batch.observations.reshape(-1, self.argus.observation_dim)
                actions = batch.actions.reshape(-1, self.argus.action_dim)
                next_observations = batch.next_observations.reshape(-1, self.argus.observation_dim)
                rewards = batch.rewards.reshape(-1, 1)
                dones = batch.dones.reshape(-1, 1)
                loss = {}
                if update_energy:
                    energy_loss_info = self.energy_train(
                        iql_tau=self.argus.iql_tau, observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(energy_loss_info)
                if update_behavior:
                    energy_loss_info = self.behavior_flow_train(observations=observations, actions=actions)
                    loss.update(energy_loss_info)
                if update_flow:
                    flow_loss_info = self.grpo_value_train(
                        observations=observations[:-1], actions=actions[:-1], next_observations=observations[1:], next_actions=actions[1:], multiple_actions=self.argus.adv_rl_multiple_actions)
                    loss.update(flow_loss_info)
                    flow_loss_info = self.grpo_policy_train(observations=observations[:-1], actions=actions[:-1], next_observations=observations[1:], next_actions=actions[1:],
                                                            expectile=self.argus.gfpo_expectile)
                    loss.update(flow_loss_info)

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        wandb_infos = {}
                        wandb_infos.update(loss)
                        # wandb_infos.update({"diffusion_loss": loss})
                        for info_key, info_val in wandb_infos.items():
                            wandb_infos[info_key] = info_val
                        wandb.log(wandb_infos, step=self.step)


                if self.step % 10000 == 0 and update_flow:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def full_rl_like_value_train(self, observations, actions, next_observations, rewards, dones):
        x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        flow_observations = torch.cat([observations, x_t], dim=-1)
        flow_next_observations = torch.cat([next_observations, x_t], dim=-1)
        Q_loss, V_loss, loss_info = self.flow_energy_model.loss(
            tau=self.argus.iql_tau, observations=flow_observations, actions=dx_dt, normed_actions=normed_dx_dt,
            next_observations=flow_next_observations, t=t, next_actions=None, rewards=rewards, dones=dones,
            fake_actions=None, fake_next_actions=None)
        V_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow_energy_model.v.parameters(), max_norm=10.0)
        self.fv_v_optimizer.step()
        self.fv_v_optimizer.zero_grad()
        Q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow_energy_model.q.parameters(), max_norm=10.0)
        self.fv_optimizer.step()
        self.fv_optimizer.zero_grad()
        self.flow_energy_model.q_ema()
        return loss_info

    def full_rl_like_policy_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        cos_sim = F.cosine_similarity(pred_u, behavior_u.detach(), dim=-1)
        divergence = 1 - cos_sim.unsqueeze(dim=-1)
        normed_pred_u = pred_u / torch.norm(pred_u, dim=-1, keepdim=True)
        pred_Q = self.flow_energy_model.q_target(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        pred_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_pred_u], dim=-1), t=t)
        loss = -(pred_Q - pred_V)

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def full_rl_like_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = 40
        except:
            print(colored("Load critic checkpoint unsuccessful !!!!", "red"))
            raise NotImplementedError
        for epoch in range(epoch_offset, num_epochs):
            if self.argus.debug_mode:
                update_flow = True
                update_flow_value = True
                update_flow_policy = True
            else:
                update_flow = False if epoch < 40 else True
                update_flow_value = True
                update_flow_policy = True
            for step in range(num_steps_per_epoch):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                observations = batch.observations.reshape(-1, self.argus.observation_dim)
                actions = batch.actions.reshape(-1, self.argus.action_dim)
                next_observations = batch.next_observations.reshape(-1, self.argus.observation_dim)
                rewards = batch.rewards.reshape(-1, 1)
                dones = batch.dones.reshape(-1, 1)
                loss = {}
                if update_flow:
                    flow_value_loss_info = self.full_rl_like_value_train(observations=observations, actions=actions, next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(flow_value_loss_info)
                    flow_policy_loss_info = self.full_rl_like_policy_train(observations=observations, actions=actions)
                    loss.update(flow_policy_loss_info)

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        wandb_infos = {}
                        wandb_infos.update(loss)
                        # wandb_infos.update({"diffusion_loss": loss})
                        for info_key, info_val in wandb_infos.items():
                            wandb_infos[info_key] = info_val
                        wandb.log(wandb_infos, step=self.step)


                if (self.step+1) % 10000 == 0 and update_flow_policy:
                    self.eval()

                if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                    self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']}/{self.best_model_info['performance_std']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def eval(self):
        print(f"Perform evaluation on guidance_scale ......")
        eval_results = parallel_d4rl_eval_score_function_version(
            argus=self.argus, dataset=self.dataset, model=self.model,
            critic=self.flow_energy_model, guidance_scale=1.0, eval_episodes=self.argus.eval_episodes, ddim_sample=True)
        for key, val in eval_results.items():
            if "ave_score" in key:
                if self.best_model_info["performance"] < val:
                    self.best_model_info["performance"] = val
                    for key__, val__ in eval_results.items():
                        if "std_score" in key__:
                            self.best_model_info["performance_std"] = val__
                    data = {
                        'guidance_scale': self.best_model_info["guidance_scale"],
                        'performance': self.best_model_info["performance"],
                        'step': self.step,
                        'flow': self.model.state_dict(),
                    }
                    savepath = os.path.join(self.get_detailed_save_path(), 'best_flow_checkpoint')
                    os.makedirs(savepath, exist_ok=True)
                    torch.save(data, os.path.join(savepath, f'flow_{self.step}.pt'))
                    print(colored(f'[ utils/training ] Saved best model to {savepath}', color="green"))
        if self.wandb_log:
            wandb.log(eval_results, step=self.step)

    def save_critic_checpoint(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {}
        if self.argus.critic_type in [CriticType.iql, CriticType.ciql]:
            data.update({
                'step': self.step,
                'q': self.energy_model.q.state_dict(),
                'q_target': self.energy_model.q_target.state_dict(),
                'v': self.energy_model.v.state_dict(),
                'behavior_flow': self.behavior_flow.state_dict(),
            })
        elif self.argus.critic_type in [CriticType.isql]:
            data.update({
                'step': self.step,
                'q': self.energy_model.q.state_dict(),
                'q_target': self.energy_model.q_target.state_dict(),
                'behavior_flow': self.behavior_flow.state_dict(),
            })
        elif self.argus.critic_type in [CriticType.cql]:
            pass
        else:
            raise ValueError(f"Critic type {self.argus.critic_type} not supported")
        savepath = os.path.join(self.get_detailed_save_path(), 'critic_checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        torch.save(data, os.path.join(savepath, f'critic_{self.step}.pt'))
        torch.save(data, os.path.join(savepath, 'critic.pt'))
        print(colored(f'[ utils/training ] Saved model to {savepath}', color="green"))

    def get_checkpoint_step(self, loadpath, step_offset):
        checkpoints_list = os.listdir(loadpath)
        checkpoints_step = []
        for checkpoint in checkpoints_list:
            if "_" in checkpoint:
                checkpoints_step.append(int(checkpoint.split("_")[-1].split(".")[0]))
        checkpoints_step.sort()
        if step_offset=="latest":
            return checkpoints_step[-1]
        else:
            return checkpoints_step[0] + step_offset

    def load_critic_checkpoint(self, loadpath, step_offset="latest"):
        step = self.get_checkpoint_step(
            loadpath=os.path.join(loadpath, 'critic_checkpoint'), step_offset=step_offset)
        loadpath = os.path.join(loadpath, f'critic_checkpoint/critic_{step}.pt')
        data = torch.load(loadpath)
        print(colored(f" From {loadpath} Load Critic Model with saved train step {step} Success !!!", color="green"))
        if self.argus.critic_type in [CriticType.iql, CriticType.ciql]:
            self.energy_model.q.load_state_dict(data['q'])
            self.energy_model.q_target.load_state_dict(data['q_target'])
            self.energy_model.v.load_state_dict(data['v'])
            if self.argus.rl_mode not in [RLTrainMode.flow_constrained_rl5]:
                if self.argus.dataset in self.argus.adroit_dataset:
                    pass
                else:
                    self.behavior_flow.load_state_dict(data['behavior_flow'])
                    print(colored(f" From {loadpath} Load Behavior Model with saved train step {step} Success !!!", color="green"))
        elif self.argus.critic_type in [CriticType.isql]:
            self.energy_model.q.load_state_dict(data['q'])
            self.energy_model.q_target.load_state_dict(data['q_target'])
            self.behavior_flow.load_state_dict(data['behavior_flow'])
            print(colored(f" From {loadpath} Load Behavior Model with saved train step {step} Success !!!", color="green"))
        elif self.argus.critic_type in [CriticType.cql]:
            self.energy_model.q.load_state_dict(data['q'])
            self.energy_model.q_target.load_state_dict(data['q_target'])
            if self.argus.rl_mode not in [RLTrainMode.flow_constrained_rl5]:
                if self.argus.dataset in self.argus.adroit_dataset:
                    pass
                else:
                    self.behavior_flow.load_state_dict(data['behavior_flow'])
                    print(colored(f" From {loadpath} Load Behavior Model with saved train step {step} Success !!!", color="green"))
        elif self.argus.critic_type in [CriticType.cql]:
            pass
        else:
            raise ValueError(f"Critic type {self.argus.critic_type} not supported")

    def load_best_flow_checkpoint(self, loadpath, step_offset="latest"):
        step = self.get_checkpoint_step(
            loadpath=os.path.join(loadpath, 'best_flow_checkpoint'), step_offset=step_offset)
        loadpath = os.path.join(loadpath, f'best_flow_checkpoint/flow_{step}.pt')
        data = torch.load(loadpath)
        print(colored(f" From {loadpath} Load Flow Model with saved train step {step} Success !!!", color="green"))
        if self.argus.critic_type in [CriticType.iql, CriticType.ciql, CriticType.isql]:
            self.model.load_state_dict(data['flow'])
            best_per = data["performance"]
            print(colored(f"Best Flow Model Performance: {best_per}", color="green"))
        elif self.argus.critic_type in CriticType.cql:
            pass
        else:
            raise ValueError(f"Critic type {self.argus.critic_type} not supported")




