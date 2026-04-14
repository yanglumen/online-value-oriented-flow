import os.path
import wandb
import torch
import random
import torch.nn as nn
import numpy as np
from models.expectile_forward_process import sample_weighted_interpolated_points
from trainer.trainer_util import (
    batch_to_device,
)
from termcolor import colored
from config.dict2class import obj2dict
from config.expectile_flow_hyperparameter import *
import matplotlib.pyplot as plt
from evaluation.sequential_eval import parallel_d4rl_eval_score_function_version

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
    def __init__(self, argus, model, energy_model, expectile_q, expectile_v, dataset):
        self.argus = argus
        self.model = model
        self.model.to(self.argus.device)
        self.energy_model = energy_model
        self.energy_model.to(self.argus.device)
        self.expectile_q = expectile_q
        self.expectile_q.to(self.argus.device)
        self.expectile_v = expectile_v
        self.expectile_v.to(self.argus.device)
        self.dataset = dataset
        self.device = argus.device
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=argus.lr)
        self.eq_optimizer = torch.optim.Adam(self.expectile_q.parameters(), lr=argus.lr)
        self.ev_optimizer = torch.optim.Adam(self.expectile_v.parameters(), lr=argus.lr)
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
        if self.wandb_log:
            wandb.init(name=self.wandb_exp_name, group=self.wandb_exp_group, project=self.wandb_project_name, config=obj2dict(self.argus))

    def get_detailed_save_path(self):
        return os.path.join(
            self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.current_exp_label)

    def expectile_flow_train(self, observations, actions):
        s_x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        # pred_dx_dt = self.model(s_x_t, t)
        # normed_pred_dx_dt = pred_dx_dt / torch.norm(pred_dx_dt, dim=-1, keepdim=True)
        # flow_loss = -self.expectile_q(torch.cat([s_x_t, normed_pred_dx_dt], dim=1))

        with torch.no_grad():
            normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
            eq_value = self.expectile_q(x=torch.cat([s_x_t, normed_dx_dt], dim=1), t=t)
            ev_value = self.expectile_v(x=s_x_t, t=t)
            u = eq_value - ev_value
            weights = torch.where(u > 0, self.argus.expectile_flow_tau, (1 - self.argus.expectile_flow_tau)).detach()
        pred_dx_dt = self.model(s_x_t, t)
        flow_loss = weights * torch.norm(pred_dx_dt - dx_dt.detach(), dim=-1, keepdim=True)
        flow_loss = flow_loss.mean()
        self.optimizer.zero_grad()
        flow_loss.backward()
        self.optimizer.step()
        self.eq_optimizer.zero_grad()
        return {"flow_loss": flow_loss.detach().cpu().numpy().item()}

    def expectile_rl_flow_train(self, observations, actions):
        s_x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        pred_dx_dt = self.model(s_x_t, t)
        # todo corresponding to the strategy that uses normed direction dxdt to train rl-based flow
        normed_pred_dx_dt = pred_dx_dt / torch.norm(pred_dx_dt, dim=-1, keepdim=True)
        flow_loss = -self.expectile_q(x=torch.cat([s_x_t, normed_pred_dx_dt], dim=-1), t=t)
        # flow_loss = -(self.expectile_q(x=torch.cat([s_x_t, normed_pred_dx_dt], dim=-1), t=t)-self.expectile_v(x=s_x_t, t=t))
        # todo ciql_expectile_rl_time_advantage
        # normed_pred_dx_dt = pred_dx_dt / torch.norm(pred_dx_dt, dim=-1, keepdim=True)
        # adv = self.expectile_q(x=torch.cat([s_x_t, normed_pred_dx_dt], dim=1), t=t) - self.expectile_v(x=s_x_t, t=t)
        # flow_loss = -adv
        # todo corresponding to the strategy that directly uses direction dxdt to train rl-based flow
        # flow_loss = -self.expectile_q(x=torch.cat([s_x_t, pred_dx_dt], dim=1), t=t)
        flow_loss = flow_loss.mean()
        self.optimizer.zero_grad()
        flow_loss.backward()
        self.optimizer.step()
        self.eq_optimizer.zero_grad()
        return {"flow_loss": flow_loss.detach().cpu().numpy().item()}

    def flow_train(self, observations, actions):
        if self.argus.flow_guided_mode == FlowGuidedMode.normal:
            if self.argus.expectile_type == ExpectileMode.expectile:
                flow_loss_info = self.expectile_flow_train(observations=observations, actions=actions)
            elif self.argus.expectile_type == ExpectileMode.expectile_rl:
                flow_loss_info = self.expectile_rl_flow_train(observations=observations, actions=actions)
            else:
                raise ValueError('expectile_type wrong!')
            return flow_loss_info
        raise NotImplementedError

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

    def expectile_loss(self, tau, u):
        weight = torch.where(u > 0, tau, (1 - tau))
        out = weight * torch.pow(u, 2)
        # out = torch.abs(tau - (u < 0).float()).detach() * torch.pow(u, 2)
        return out.mean()

    def expectile_value_train(self, observations, actions):
        s_x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        eq_value = self.expectile_q(x=torch.cat([s_x_t, normed_dx_dt], dim=1), t=t)
        ev_value = self.expectile_v(x=s_x_t, t=t)
        with torch.no_grad():
            energy = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        eq_loss = self.loss_fn(eq_value, energy.detach())
        ev_loss = self.expectile_loss(tau=self.argus.expectile_func_tau, u=eq_value.detach() - ev_value)

        self.eq_optimizer.zero_grad()
        eq_loss.backward()
        self.eq_optimizer.step()
        self.ev_optimizer.zero_grad()
        ev_loss.backward()
        self.ev_optimizer.step()
        return {"eq_loss": eq_loss.detach().cpu().numpy().item(),
                "ev_loss": ev_loss.detach().cpu().numpy().item(),
                "eq_mean_value": eq_value.mean().detach().cpu().numpy().item(),
                "ev_mean_value": ev_value.mean().detach().cpu().numpy().item(),
                }

    def generate_noise_direction(self, dx_dt, cos_similarity_threshold=0.95, scale_threshold=0.1, mode="naive"):
        random_gaussian_direction = torch.randn_like(dx_dt, device=self.argus.device).clip(-1, 1)
        if mode == "naive":
            noise = random_gaussian_direction * scale_threshold
        elif mode == "cos_similarity":
            theta = torch.acos(torch.tensor(cos_similarity_threshold, device=self.argus.device))
            theta = torch.rand((len(dx_dt), 1), device=self.argus.device) * theta
            normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
            noise = random_gaussian_direction - torch.sum(random_gaussian_direction * normed_dx_dt, dim=-1, keepdim=True)-normed_dx_dt
            noise = noise / torch.norm(noise, dim=-1, keepdim=True)
            noise = torch.cos(theta) * normed_dx_dt + torch.sin(theta) * noise
        else:
            raise ValueError('mode wrong!')
        return noise

    def expectile_rl_value_train(self, observations, actions):
        s_x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        # todo pure conservative optimization: only focus on the exist points from dataset
        # random_direction = torch.randn_like(dx_dt, device=self.argus.device)
        # normed_random_direction = random_direction / torch.norm(random_direction, dim=-1, keepdim=True)
        # normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        # eq_value = self.expectile_q(
        #     x=torch.cat([torch.cat([s_x_t, normed_dx_dt], dim=-1),
        #                  torch.cat([s_x_t, normed_random_direction], dim=-1)], dim=0),
        #     t=torch.cat([t, t], dim=0))
        # with torch.no_grad():
        #     energy = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        #     energy = torch.cat([energy, torch.zeros_like(energy)], dim=0)
        # eq_loss = self.loss_fn(eq_value, energy.detach())
        # self.eq_optimizer.zero_grad()
        # eq_loss.backward()
        # self.eq_optimizer.step()
        # return {"eq_loss": eq_loss.detach().cpu().numpy().item(),
        #         "eq_mean_value": eq_value.mean().detach().cpu().numpy().item()}

        #  todo ciql_expectile_rl_time_censervative
        #   add direction noise will improve the performance based on pure conservative optimization
        random_direction = torch.randn_like(dx_dt, device=self.argus.device)
        normed_random_direction = random_direction / torch.norm(random_direction, dim=-1, keepdim=True)
        # noise = torch.randn_like(dx_dt, device=self.argus.device).clip(-1, 1) * self.argus.noise_scale_threshold
        # action_noise_direction = noise + dx_dt
        # action_noise_direction = self.generate_noise_direction(
        #     dx_dt=dx_dt, scale_threshold=self.argus.noise_scale_threshold, cos_similarity_threshold=self.argus.noise_similarity_threshold)
        # normed_action_noise_direction = action_noise_direction / torch.norm(action_noise_direction, dim=-1, keepdim=True)
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        eq_value = self.expectile_q(
            x=torch.cat([torch.cat([s_x_t, normed_dx_dt], dim=-1),
                         # torch.cat([s_x_t, normed_action_noise_direction], dim=-1),
                         torch.cat([s_x_t, normed_random_direction], dim=-1)], dim=0),
            t=torch.cat([t, t], dim=0))
        with torch.no_grad():
            # noise_direction_energy = self.energy_model.get_scaled_q(
            #     obs=observations, act=actions+noise, scale=self.argus.energy_scale)
            energy = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
            energy = torch.cat([energy, torch.zeros_like(energy)], dim=0)
        eq_loss = self.loss_fn(eq_value, energy.detach())
        self.eq_optimizer.zero_grad()
        eq_loss.backward()
        self.eq_optimizer.step()
        # ev_value = self.expectile_v(x=s_x_t, t=t)
        # ev_loss = self.loss_fn(ev_value, eq_value[:len(s_x_t)].detach())
        # self.ev_optimizer.zero_grad()
        # ev_loss.backward()
        # self.ev_optimizer.step()
        return {"eq_loss": eq_loss.detach().cpu().numpy().item(),
                "eq_mean_value": eq_value.mean().detach().cpu().numpy().item(),
                # "ev_loss": ev_loss.detach().cpu().numpy().item(),
                # "ev_mean_value": ev_value.mean().detach().cpu().numpy().item(),
                }

        #  todo ciql_expectile_rl_time_advantage
        #   sample random direction to calculate the V value
        # noise_sample = 3
        # x_t = s_x_t[:, self.argus.observation_dim:]
        # s_x_t = s_x_t.unsqueeze(dim=1).repeat(1, noise_sample+1, 1)
        # t = t.unsqueeze(dim=1).repeat(1, noise_sample+1, 1)
        # random_direction = torch.randn((dx_dt.shape[0], noise_sample, dx_dt.shape[-1]), device=self.argus.device)
        # random_direction = torch.cat([random_direction, dx_dt.unsqueeze(dim=1)], dim=1)
        # normed_random_direction = random_direction / torch.norm(random_direction, dim=-1, keepdim=True)
        # random_x_1 = s_x_t[:, :,self.argus.observation_dim:] + normed_random_direction * torch.norm(dx_dt.unsqueeze(dim=1), dim=-1, keepdim=True) * (1-t)
        # eq_value = self.expectile_q(
        #     x=torch.cat([s_x_t, normed_random_direction], dim=-1), t=t)
        # with torch.no_grad():
        #     energy = self.energy_model.get_scaled_q(
        #         obs=observations.unsqueeze(dim=1).repeat(1, noise_sample+1, 1), act=random_x_1, scale=self.argus.energy_scale)
        # eq_loss = self.loss_fn(eq_value, energy.detach())
        # self.eq_optimizer.zero_grad()
        # eq_loss.backward()
        # self.eq_optimizer.step()
        # ev_value = self.expectile_v(x=torch.cat([observations, x_t], dim=-1), t=t[:,0,:])
        # ev_loss = self.loss_fn(ev_value, torch.mean(energy, dim=1).detach())
        # self.ev_optimizer.zero_grad()
        # ev_loss.backward()
        # self.ev_optimizer.step()
        # return {"eq_loss": eq_loss.detach().cpu().numpy().item(),
        #         "ev_loss": ev_loss.detach().cpu().numpy().item(),
        #         "eq_mean_value": eq_value.mean().detach().cpu().numpy().item(),
        #         "ev_mean_value": ev_value.mean().detach().cpu().numpy().item(),
        #         }

        # todo ciql_expectile_rl_time_noise_dxdt_1
        #  different from the above strategy where we only use the dierection of dxdt, here we want to direct fit
        #  the dxdt rather than the normed dxdt
        # random_direction = torch.randn_like(dx_dt, device=self.argus.device)
        # action_noise_direction = torch.randn_like(dx_dt, device=self.argus.device).clip(-1, 1) * 0.005 + dx_dt
        # eq_value = self.expectile_q(
        #     x=torch.cat([torch.cat([s_x_t, dx_dt], dim=-1),
        #                  torch.cat([s_x_t, action_noise_direction], dim=-1),
        #                  torch.cat([s_x_t, random_direction], dim=-1)], dim=0),
        #     t=torch.cat([t, t, t], dim=0))
        # with torch.no_grad():
        #     energy = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        #     noise_direction_energy = self.energy_model.get_scaled_q(
        #         obs=observations, act=s_x_t[:,self.argus.observation_dim:]+action_noise_direction, scale=self.argus.energy_scale)
        #     target_energy = torch.cat([energy, noise_direction_energy, torch.zeros_like(energy)], dim=0)
        # eq_loss = self.loss_fn(eq_value, target_energy.detach())
        # self.eq_optimizer.zero_grad()
        # eq_loss.backward()
        # self.eq_optimizer.step()
        # return {"eq_loss": eq_loss.detach().cpu().numpy().item(),
        #         "eq_mean_value": eq_value.mean().detach().cpu().numpy().item()}

        # todo without conservative optimization, the expectile rl method can only induces poor performance
        # random_direction = torch.randn_like(dx_dt, device=self.argus.device)
        # random_action = s_x_t[:, self.argus.observation_dim:] + random_direction
        # normed_random_dx_dt = random_direction / torch.norm(random_direction, dim=-1, keepdim=True)
        # normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        # eq_value = self.expectile_q(
        #     x=torch.cat([torch.cat([s_x_t, normed_random_dx_dt], dim=-1),
        #                  torch.cat([s_x_t, normed_dx_dt], dim=-1)], dim=0),
        #     t=torch.cat([t, t], dim=0))
        # ev_value = self.expectile_v(
        #     x=torch.cat([s_x_t, s_x_t], dim=0),
        #     t=torch.cat([t, t], dim=0))
        # with torch.no_grad():
        #     energy = self.energy_model.get_scaled_q(
        #         obs=torch.cat([observations, observations], dim=0),
        #         act=torch.cat([random_action, actions], dim=0), scale=self.argus.energy_scale)
        # eq_loss = self.loss_fn(eq_value, energy.detach())
        # ev_loss = self.expectile_loss(tau=self.argus.expectile_func_tau, u=eq_value.detach() - ev_value)
        #
        # self.eq_optimizer.zero_grad()
        # eq_loss.backward()
        # self.eq_optimizer.step()
        # self.ev_optimizer.zero_grad()
        # ev_loss.backward()
        # self.ev_optimizer.step()
        # return {"eq_loss": eq_loss.detach().cpu().numpy().item(),
        #         "ev_loss": ev_loss.detach().cpu().numpy().item(),
        #         "eq_mean_value": eq_value.mean().detach().cpu().numpy().item(),
        #         "ev_mean_value": ev_value.mean().detach().cpu().numpy().item(),
        #         }

        # todo original
        # random_direction = torch.randn_like(dx_dt, device=self.argus.device)
        # # normed_random_direction = random_direction / torch.norm(random_direction, dim=-1, keepdim=True)
        # # random_direction = normed_random_direction * torch.norm(dx_dt, dim=-1, keepdim=True)
        # random_action = actions - random_direction
        # random_s_x_t, random_t, random_dx_dt, _ = sample_weighted_interpolated_points(
        #     argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
        #     energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.specific_start_end,
        #     x_0=random_action,
        # )
        # normed_random_dx_dt = random_dx_dt / torch.norm(random_dx_dt, dim=-1, keepdim=True)
        # normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        # eq_value = self.expectile_q(
        #     x=torch.cat([torch.cat([s_x_t, normed_random_dx_dt], dim=-1),
        #                  torch.cat([s_x_t, normed_dx_dt], dim=-1)], dim=0),
        #     t=torch.cat([random_t, t], dim=0))
        # ev_value = self.expectile_v(
        #     x=torch.cat([random_s_x_t, s_x_t], dim=0), t=torch.cat([random_t, t], dim=0))
        # with torch.no_grad():
        #     energy = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        # eq_loss = self.loss_fn(eq_value, torch.cat([energy.detach(), energy.detach()], dim=0))
        # ev_loss = self.expectile_loss(tau=self.argus.expectile_func_tau, u=eq_value.detach() - ev_value)
        #
        # self.eq_optimizer.zero_grad()
        # eq_loss.backward()
        # self.eq_optimizer.step()
        # self.ev_optimizer.zero_grad()
        # ev_loss.backward()
        # self.ev_optimizer.step()
        # return {"eq_loss": eq_loss.detach().cpu().numpy().item(),
        #         "ev_loss": ev_loss.detach().cpu().numpy().item(),
        #         "eq_mean_value": eq_value.mean().detach().cpu().numpy().item(),
        #         "ev_mean_value": ev_value.mean().detach().cpu().numpy().item(),
        #         }

    def value_train(self, observations, actions):
        if self.argus.expectile_type == ExpectileMode.expectile:
            energy_loss_info = self.expectile_value_train(observations=observations, actions=actions)
        elif self.argus.expectile_type == ExpectileMode.expectile_rl:
            energy_loss_info = self.expectile_rl_value_train(observations=observations, actions=actions)
        else:
            raise ValueError('expectile_type wrong!')
        return energy_loss_info

    def guided_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            # self.load_best_flow_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = 40
        except:
            epoch_offset = 0
        for epoch in range(epoch_offset, num_epochs):
            for step in range(num_steps_per_epoch):
                if self.argus.debug_mode:
                    update_energy = True
                    update_flow = True
                else:
                    update_energy = True if epoch < 40 else False
                    update_flow = False if epoch < 40 else True
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
                if update_flow:
                    energy_loss_info = self.value_train(observations=observations, actions=actions)
                    loss.update(energy_loss_info)
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

                if (self.step+1) % 5000 == 0 and update_flow:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def eval(self):
        print(f"Perform evaluation on guidance_scale ......")
        eval_results = parallel_d4rl_eval_score_function_version(
            argus=self.argus, dataset=self.dataset, model=self.model,
            critic=self.energy_model, guidance_scale=1.0,
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
                # 'eq': self.expectile_q.state_dict(),
                # 'ev': self.expectile_v.state_dict(),
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
            # self.expectile_q.load_state_dict(data['eq'])
            # self.expectile_v.load_state_dict(data['ev'])
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