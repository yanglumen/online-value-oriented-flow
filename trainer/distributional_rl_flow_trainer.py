import os.path
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
from evaluation.sequential_eval import parallel_d4rl_eval_score_function_version


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

    def direct_set_model(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = up_weight

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
        self.energy_model = energy_model
        self.energy_model.to(self.argus.device)
        self.dataset = dataset
        self.device = argus.device
        self.loss_fn = nn.MSELoss()
        self.ema = EMA(argus.ema_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=argus.lr)
        self.bf_optimizer = torch.optim.Adam(self.behavior_flow.parameters(), lr=argus.lr)
        if self.argus.rl_mode in [RLTrainMode.flow_constrained_rl2, RLTrainMode.dist_flow_constrained_rl]:
            self.q_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
            self.fv_v_optimizer = torch.optim.Adam(self.flow_energy_model.flow_v.parameters(), lr=argus.lr)
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl:
            self.q_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
        elif self.argus.rl_mode == RLTrainMode.use_rl_q:
            self.q_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
            # self.fv_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.fv_v_optimizer = torch.optim.Adam(self.flow_energy_model.flow_v.parameters(), lr=argus.lr)
        else:
            self.fv_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.fv_v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
            self.fv_v_optimizer = torch.optim.Adam(self.flow_energy_model.flow_v.parameters(), lr=argus.lr)
        self.energy_q_optimizer = torch.optim.Adam(self.energy_model.q.parameters(), lr=argus.lr)
        self.energy_v_optimizer = torch.optim.Adam(self.energy_model.v.parameters(), lr=argus.lr)
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
        self.ema.direct_set_model(ma_model=self.target_model, current_model=self.model)
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

    def rl_like_flow_value_train(self, observations, actions, next_observations, rewards, dones, GRPO_mode_update=True):
        if GRPO_mode_update:
            Q_loss, V_loss, flow_V_loss, loss_info = self.flow_energy_model.loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
            if Q_loss is not None:
                self.q_optimizer.zero_grad()
                Q_loss.backward()
                self.q_optimizer.step()
                self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
            if V_loss is not None:
                self.v_optimizer.zero_grad()
                V_loss.backward()
                self.v_optimizer.step()
            if flow_V_loss is not None:
                self.fv_v_optimizer.zero_grad()
                flow_V_loss.backward()
                self.fv_v_optimizer.step()
            return loss_info
            # num_samples = len(observations)
            # multiple_next_observation = next_observations.unsqueeze(1).repeat(1, self.argus.grpo_group_size, 1)
            # x = torch.randn(num_samples, self.argus.grpo_group_size, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            # dt = 1.0 / self.argus.flow_step
            # multi_divergence, multi_dxdt, multi_x = [], [], [x]
            # # total_loss = 0
            # step_index = 0
            # time_start, time_end, steps = 0, 1.0, self.argus.flow_step
            # for t in np.linspace(time_start, time_end, steps):
            #     t_tensor = torch.full((num_samples, self.argus.grpo_group_size, 1), t, dtype=torch.float32, device=self.argus.device)
            #     with torch.no_grad():
            #         _, flow_u, _ = self.model(torch.cat([multiple_next_observation, x], dim=-1), t_tensor)
            #         behavior_u = self.behavior_flow(x=torch.cat([multiple_next_observation, x], dim=-1), t=t_tensor)
            #     multi_dxdt.append(flow_u.detach())
            #     divergence = torch.norm(flow_u - behavior_u.detach(), dim=-1, keepdim=True)
            #     multi_divergence.append(divergence.detach())
            #     # flow_V = self.flow_energy_model.flow_v(torch.cat([multiple_observation, x, flow_u], dim=-1))
            #     x = x + flow_u * dt
            #     multi_x.append(x)
            #     x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            #     step_index += 1
            #     # if step_index >= self.argus.flow_step:
            #     #     target_flow_V = -divergence + self.energy_model.get_scaled_q(obs=multiple_observation, act=x, scale=self.argus.energy_scale)
            #     # else:
            #     #     target_flow_V = -divergence + self.flow_energy_model.target_flow_v(torch.cat([multiple_observation, x, flow_u], dim=-1))
            #     # flow_V_loss = self.loss_fn(flow_V, target_flow_V.detach())
            #     # total_loss += flow_V_loss
            # min_Q = self.energy_model.get_scaled_q(obs=multiple_next_observation, act=x, scale=self.argus.energy_scale)
            # flow_v_loss = 0
            # for step_i in range(len(multi_divergence)):
            #     target_flow_V = - self.argus.divergence_coef * torch.sum(torch.cat(multi_divergence[step_i:], dim=-1), dim=-1, keepdim=True) + min_Q.detach()
            #     flow_V = self.flow_energy_model.flow_v(torch.cat([multi_x[step_i], actions.unsqueeze(1).repeat(1, self.argus.grpo_group_size, 1), multiple_next_observation], dim=-1))
            #     flow_v_loss += F.mse_loss(flow_V, target_flow_V.detach()) / len(multi_divergence)
            #
            # self.fv_v_optimizer.zero_grad()
            # flow_v_loss.backward()
            # self.fv_v_optimizer.step()
            # return {
            #     "flow_V": flow_V.mean().detach().cpu().numpy().item(),
            #     "flow_v_value_loss": flow_v_loss.detach().cpu().numpy().item(),
            # }
        else:
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

    def rl_like_flow_policy_train(self, observations, actions, next_observations, GRPO_mode_update=True):
        loss_info = {}
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        if GRPO_mode_update:
            num_samples = len(observations)
            multiple_next_observations = next_observations.unsqueeze(1).repeat(1, self.argus.grpo_group_size, 1)
            x = torch.randn(num_samples, self.argus.grpo_group_size, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            dt = 1.0 / self.argus.flow_step
            multi_value_loss, multi_ratio, multi_divergence, multi_entropy = [], [], [], []
            total_divergence = 0
            step_index = 0
            time_start, time_end, steps = 0, 1.0, self.argus.flow_step
            for t in np.linspace(time_start, time_end, steps):
                t_tensor = torch.full((num_samples, self.argus.grpo_group_size, 1), t, dtype=torch.float32, device=self.argus.device)
                with torch.no_grad():
                    sampled_u, flow_target_u, log_pi_old = self.target_model.sample_action(torch.cat([multiple_next_observations, x], dim=-1), t_tensor)
                    behavior_u = self.behavior_flow(x=torch.cat([multiple_next_observations, x], dim=-1), t=t_tensor)
                log_pi, flow_u, entropy_u = self.model.log_prob_mu_entropy(torch.cat([multiple_next_observations, x], dim=-1), t_tensor, a=sampled_u)
                divergence = torch.norm(flow_u - behavior_u.detach(), dim=-1, keepdim=True)
                multi_divergence.append(divergence)
                total_divergence += divergence
                ratio = torch.exp(log_pi - log_pi_old.detach())
                multi_ratio.append(ratio)
                x = x + flow_target_u * dt
                value_loss = self.argus.divergence_coef*multi_divergence[-1] - self.flow_energy_model.flow_v(
                    torch.cat([x, actions.unsqueeze(1).repeat(1, self.argus.grpo_group_size, 1), multiple_next_observations], dim=-1))
                multi_value_loss.append(value_loss)
                with torch.no_grad():
                    quantile_80 = torch.quantile(value_loss, 0.8, dim=1, keepdim=True)
                    entropy_weights = torch.where(value_loss < quantile_80, torch.tensor(-1.0, device=value_loss.device), torch.tensor(1.0, device=value_loss.device))
                    # entropy_weights = torch.softmax(torch.mean(value_loss, dim=1, keepdim=True) - value_loss, dim=1)
                multi_entropy.append(entropy_u*entropy_weights)
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                step_index += 1
            divergence = torch.mean(total_divergence, dim=1)
            x = torch.clamp(x, min=-self.argus.max_action_val, max=self.argus.max_action_val)
            pred_Q = self.flow_energy_model.get_scaled_q(obs=multiple_next_observations, act=x, scale=self.argus.energy_scale)
            adv = pred_Q - torch.mean(pred_Q, dim=1, keepdim=True)# / (torch.std(pred_Q, dim=1, keepdim=True) + 1e-8)
            adv = adv.detach()
            flow_value_loss = torch.sum(torch.cat(multi_value_loss, dim=-1), dim=-1)
            entropy_loss = torch.sum(torch.cat(multi_entropy, dim=-1), dim=-1)
            ratio = torch.cat(multi_ratio, dim=-1)
            surr1 = torch.sum(torch.mean(ratio * adv, dim=1), dim=-1, keepdim=True)
            surr2 = torch.sum(torch.mean(torch.clamp(ratio, 1 - self.argus.ppo_clip_rate, 1 + self.argus.ppo_clip_rate+0.08) * adv, dim=1), dim=-1, keepdim=True)
            loss = -torch.min(surr1, surr2).mean() + flow_value_loss.mean() + entropy_loss.mean()
            loss_info.update({
                "entropy_loss": entropy_loss.mean().detach().cpu().numpy().item(),
            })
        else:
            pi, flow_u = self.model.log_prob_and_mu(x=torch.cat([observations, x_t], dim=-1), t=t, a=dx_dt)
            pi_old, behavior_u = self.behavior_flow.log_prob_and_mu(x=torch.cat([observations, x_t], dim=-1), t=t, a=dx_dt)
            ratio = torch.exp(pi - pi_old)
            divergence = torch.norm(flow_u - behavior_u, dim=-1, keepdim=True)
            with torch.no_grad():
                pred_Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, flow_u], dim=-1), t=t)
                normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
                pred_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
                adv = pred_Q - pred_V
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.argus.ppo_clip_rate, 1 + self.argus.ppo_clip_rate) * adv
            loss = -torch.min(surr1, surr2)

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_v_optimizer.zero_grad()
        if self.step % 5 == 0:
            self.ema.update_model_average(self.target_model, self.model)
        loss_info.update({
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        })
        return loss_info
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

    def flow_train(self, observations, actions, next_observations, rewards, dones, update_flow_value, update_flow_policy):
        self.model.train()
        loss = {}
        if update_flow_value:
            loss_info = self.rl_like_flow_value_train(observations=observations, actions=actions, next_observations=next_observations, rewards=rewards, dones=dones)
            loss.update(loss_info)
        if update_flow_policy:
            loss_info = self.rl_like_flow_policy_train(observations=observations, actions=actions, next_observations=next_observations)
            loss.update(loss_info)
        return loss

    def energy_train(self, iql_tau, observations, actions, next_observations, rewards, dones, next_actions=None):
        q_loss, v_loss, loss_info = self.energy_model.loss(
            tau=iql_tau, observations=observations, actions=actions, next_observations=next_observations,
            next_actions=next_actions, rewards=rewards, dones=dones, fake_actions=None,
            fake_next_actions=None)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.energy_model.v.parameters(), max_norm=10.0)
        self.energy_v_optimizer.step()
        self.energy_v_optimizer.zero_grad()
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
        self.eval()
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
                    flow_loss_info = self.flow_train(
                        observations=observations, actions=actions, next_observations=next_observations,
                        rewards=rewards, dones=dones,
                        update_flow_value=update_flow_value, update_flow_policy=update_flow_policy)
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

    def flow_constrained_value_train(self, observations, actions, next_observations, rewards, dones):
        if self.argus.rl_mode == RLTrainMode.flow_constrained_rl2:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl2_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl:
            Q_loss, V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
        else:
            raise NotImplementedError
        self.q_optimizer.zero_grad()
        Q_loss.backward()
        self.q_optimizer.step()
        self.v_optimizer.zero_grad()
        V_loss.backward()
        self.v_optimizer.step()
        self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        return flow_loss_info

    def flow_constrained_policy_train(self, observations, actions, next_observations, next_actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=next_observations, actions=next_actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )

        pred_u = self.model(x=torch.cat([next_observations, x_t], dim=-1), t=t)
        dt = 1 / self.argus.flow_step
        x_t_plus_deltat = x_t + pred_u * dt  # (next_t - t)
        x_t_plus_deltat = torch.clip(x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        behavior_u = self.behavior_flow(x=torch.cat([next_observations, x_t], dim=-1), t=t)
        divergence = torch.norm(pred_u - behavior_u, dim=-1, keepdim=True)
        if self.argus.rl_mode == RLTrainMode.flow_constrained_rl2:
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
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
            epoch_offset = 40
        except:
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
                    flow_loss_info = self.flow_constrained_value_train(observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(flow_loss_info)
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

                if (self.step+1) % 5000 == 0 and update_flow_policy:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def dist_behavior_flow_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        v_log_prob = self.behavior_flow.log_prob(x=torch.cat([observations, x_t], dim=-1), t=t, a=dx_dt)
        loss = -v_log_prob.mean()
        self.bf_optimizer.zero_grad()
        loss.backward()
        self.bf_optimizer.step()
        return {"dist_behavior_flow_loss": loss.detach().cpu().numpy().item()}

    def dist_flow_constrained_value_train(self, observations, actions, next_observations, rewards, dones):
        if self.argus.rl_mode == RLTrainMode.flow_constrained_rl2:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl2_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.dist_flow_constrained_rl:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.dist_flow_constrained_rl_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl:
            Q_loss, V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
        else:
            raise NotImplementedError
        self.q_optimizer.zero_grad()
        Q_loss.backward()
        self.q_optimizer.step()
        self.v_optimizer.zero_grad()
        V_loss.backward()
        self.v_optimizer.step()
        self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        return flow_loss_info

    def dist_flow_constrained_policy_train(self, observations, actions, next_observations, next_actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=next_observations, actions=next_actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        if self.argus.rl_mode == RLTrainMode.flow_constrained_rl2:
            pred_u = self.model(x=torch.cat([next_observations, x_t], dim=-1), t=t)
            dt = 1 / self.argus.flow_step
            x_t_plus_deltat = x_t + pred_u * dt  # (next_t - t)
            x_t_plus_deltat = torch.clip(x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            behavior_u = self.behavior_flow(x=torch.cat([next_observations, x_t], dim=-1), t=t)
            divergence = torch.norm(pred_u - behavior_u, dim=-1, keepdim=True)
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl:
            pred_u = self.model(x=torch.cat([next_observations, x_t], dim=-1), t=t)
            dt = 1 / self.argus.flow_step
            x_t_plus_deltat = x_t + pred_u * dt  # (next_t - t)
            x_t_plus_deltat = torch.clip(x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            behavior_u = self.behavior_flow(x=torch.cat([next_observations, x_t], dim=-1), t=t)
            divergence = torch.norm(pred_u - behavior_u, dim=-1, keepdim=True)
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.v(torch.cat([x_t_plus_deltat, next_observations], dim=-1))
        elif self.argus.rl_mode == RLTrainMode.dist_flow_constrained_rl:
            pred_u, pred_u_sigma, negative_u_log_prob = self.model.mean_var_divergence(x=torch.cat([next_observations, x_t], dim=-1), t=t, a=dx_dt)
            behavior_u, behavior_u_sigma, negative_behavior_u_log_prob = self.behavior_flow.mean_var_divergence(x=torch.cat([next_observations, x_t], dim=-1), t=t, a=dx_dt)
            dt = 1 / self.argus.flow_step
            x_t_plus_deltat = x_t + pred_u * dt  # (next_t - t)
            x_t_plus_deltat = torch.clip(x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            ratio = torch.exp(negative_behavior_u_log_prob.detach() - negative_u_log_prob)
            # divergence = 2*torch.log(behavior_u_sigma / pred_u_sigma) + (pred_u_sigma.pow(2) + (pred_u - behavior_u).pow(2)) / behavior_u_sigma.pow(2) - 0.5
            divergence = torch.norm(pred_u - behavior_u, dim=-1, keepdim=True)
            divergence = divergence.sum(dim=-1, keepdim=True)
            flow_value = self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
            # with torch.no_grad():
            #     ref_x_t_plus_deltat = x_t + behavior_u * dt
            #     ref_x_t_plus_deltat = torch.clip(ref_x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
                # reference_flow_value = self.flow_energy_model.flow_v(torch.cat([ref_x_t_plus_deltat, actions, next_observations], dim=-1))
            adv = flow_value# - reference_flow_value.detach()
            adv = (adv - adv.mean()) / ((adv.std()+1e-4))
            surr1 = ratio * adv.detach()
            surr2 = torch.clamp(ratio, 1 - self.argus.ppo_clip_rate, 1 + self.argus.ppo_clip_rate) * adv
            loss = self.argus.divergence_coef * divergence - torch.min(surr1, surr2)
        else:
            raise NotImplementedError
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "ratio": ratio.mean().detach().cpu().numpy().item(),
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def dist_flow_constrained_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = 40
        except:
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
                    pass
                if update_flow:
                    energy_loss_info = self.dist_behavior_flow_train(observations=observations, actions=actions)
                    loss.update(energy_loss_info)
                    flow_loss_info = self.dist_flow_constrained_value_train(observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(flow_loss_info)
                    flow_loss_info = self.dist_flow_constrained_policy_train(
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

                if (self.step+1) % 5000 == 0 and update_flow_policy:
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
        except:
            raise NotImplementedError
        epoch_offset = 0
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

                if (self.step+1) % 5000 == 0 and update_flow_policy:
                    self.eval()

                if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                    self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def eval(self):
        print(f"Perform evaluation on guidance_scale ......")
        eval_results = parallel_d4rl_eval_score_function_version(
            argus=self.argus, dataset=self.dataset, model=self.model,
            critic=self.flow_energy_model, guidance_scale=1.0,
            eval_episodes=self.argus.eval_episodes, ddim_sample=True)
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
        elif self.argus.critic_type in CriticType.cql:
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
            if self.argus.rl_mode in [RLTrainMode.use_rl_q, RLTrainMode.full_rl_like, RLTrainMode.flow_constrained_rl, RLTrainMode.flow_constrained_rl2]:
                self.behavior_flow.load_state_dict(data['behavior_flow'])
            elif self.argus.rl_mode in [RLTrainMode.dist_flow_constrained_rl]:
                print(colored("do not need load deterministic behavior flow policy"))
            else:
                raise ValueError(f"Critic type {self.argus.critic_type} not supported")
        elif self.argus.critic_type in CriticType.cql:
            pass
        else:
            raise ValueError(f"Critic type {self.argus.critic_type} not supported")

    def load_best_flow_checkpoint(self, loadpath, step_offset="latest"):
        step = self.get_checkpoint_step(
            loadpath=os.path.join(loadpath, 'best_flow_checkpoint'), step_offset=step_offset)
        loadpath = os.path.join(loadpath, f'best_flow_checkpoint/flow_{step}.pt')
        data = torch.load(loadpath)
        print(colored(f" From {loadpath} Load Flow Model with saved train step {step} Success !!!", color="green"))
        if self.argus.critic_type in [CriticType.iql, CriticType.ciql]:
            self.model.load_state_dict(data['flow'])
            best_per = data["performance"]
            print(colored(f"Best Flow Model Performance: {best_per}", color="green"))
        elif self.argus.critic_type in CriticType.cql:
            pass
        else:
            raise ValueError(f"Critic type {self.argus.critic_type} not supported")




