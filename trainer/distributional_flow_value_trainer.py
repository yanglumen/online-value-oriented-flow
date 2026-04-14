import os.path
import time

import wandb
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.rl_flow_forward_process import sample_weighted_interpolated_points, sample_value_interpolated_points
from trainer.trainer_util import (
    batch_to_device,
)
from termcolor import colored
from config.dict2class import obj2dict
from config.multistep_rl_flow_hyperparameter import *
from evaluation.sequential_eval import parallel_d4rl_eval_behavior_flow, parallel_d4rl_eval_adaptive_flow_step

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

class guided_flow_trainer():
    def get_detailed_save_path(self):
        return os.path.join(
            self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.current_exp_label)

    def __init__(
            self, argus, train_flow, behavior_flow, dist_flow_value, energy_model, dataset):
        self.argus = argus
        self.model = train_flow
        self.model.to(self.argus.device)
        self.behavior_flow = behavior_flow
        self.behavior_flow.to(self.argus.device)
        self.dist_flow_value = dist_flow_value
        self.dist_flow_value.to(self.argus.device)
        self.energy_model = energy_model
        self.energy_model.to(self.argus.device)
        self.dataset = dataset
        self.device = argus.device
        self.loss_fn = nn.MSELoss()
        self.ema = EMA(argus.ema_decay)
        self.flow_optimizer = torch.optim.Adam(self.model.parameters(), lr=argus.lr)
        self.bf_optimizer = torch.optim.Adam(self.behavior_flow.parameters(), lr=argus.lr)
        self.dfv_optimizer = torch.optim.Adam(self.dist_flow_value.parameters(), lr=argus.lr)
        self.energy_q_optimizer = torch.optim.Adam(self.energy_model.q.parameters(), lr=argus.lr)
        if argus.critic_type in [CriticType.iql, CriticType.ciql, CriticType.cql]:
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

    def flow_value_test(self, observations):
        with torch.no_grad():
            pred_values = self.dist_flow_value.pred_value(
                states=observations, steps=self.argus.flow_step, x_t_clip_value=self.argus.x_t_clip_value,
            )
        average_pred_values = torch.mean(pred_values)
        return average_pred_values

    def value_test(self, observations, actions):
        flow_value_test = self.flow_value_test(observations=observations)
        q_test = self.energy_model.get_scaled_q(obs=observations, act=actions)
        v_test = self.energy_model.get_scaled_v(obs=observations)
        return {"flow_value_test": flow_value_test.detach().cpu().numpy().item(),
                "q_test": q_test.mean().detach().cpu().numpy().item(),
                "v_test": v_test.mean().detach().cpu().numpy().item(),
                }

    def dist_flow_value_train(self, observations, rewards, values):
        flow_time = torch.ones_like(rewards)
        joint_observations = torch.cat([values, flow_time], dim=-1)
        # joint_next_observations = torch.cat([next_values, flow_time], dim=-1)
        x_t, t, dx_dt, weights = sample_value_interpolated_points(
            argus=self.argus, value=joint_observations, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        s_v_pred = self.dist_flow_value(torch.cat([observations, x_t[:, 0:1]], dim=-1), t)
        # next_x_t, next_t, next_dx_dt, weights = sample_value_interpolated_points(
        #     argus=self.argus, value=joint_next_observations, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        # )
        # next_s_v_pred = self.dist_flow_value(torch.cat([next_observations, next_x_t[:, 0:1]], dim=-1), next_t)
        # bellman_constrain = self.loss_fn(values, s_v_pred) + self.loss_fn(next_values, next_s_v_pred)
        loss = self.loss_fn(s_v_pred, dx_dt[:, 0:1])# + self.loss_fn(next_s_v_pred, next_dx_dt[:, 0:1]) #+ bellman_constrain
        self.dfv_optimizer.zero_grad()
        loss.backward()
        self.dfv_optimizer.step()
        return {"dist_flow_value_loss": loss.detach().cpu().numpy().item()}

    def behavior_flow_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=None, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        v_pred = self.behavior_flow(torch.cat([observations, x_t], dim=-1), t)
        loss = self.loss_fn(v_pred, dx_dt)
        self.bf_optimizer.zero_grad()
        loss.backward()
        self.bf_optimizer.step()
        return {"behavior_flow_loss": loss.detach().cpu().numpy().item()}

    def flow_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=None, weighted_samples_type=WeightedSamplesType.adaptive_step_interpolation,
        )
        v_pred = self.model(torch.cat([observations, x_t], dim=-1), t)
        loss = self.loss_fn(v_pred, dx_dt)
        self.flow_optimizer.zero_grad()
        loss.backward()
        self.flow_optimizer.step()
        return {"flow_loss": loss.detach().cpu().numpy().item()}

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

    def guided_train(self, num_epochs, num_steps_per_epoch):
        epoch_offset = 0
        for epoch in range(epoch_offset, num_epochs):
            if self.argus.debug_mode:
                update_behavior = True
                update_flow = True
                update_flow_value = True
                update_energy = True
            else:
                update_behavior = True if epoch < self.argus.update_behavior_end_epoch else False
                update_flow = False if epoch < self.argus.update_flow_start_epoch else True
                update_flow_value = False if epoch < self.argus.update_flow_start_epoch else True
                update_energy = True if epoch < 200 else False
            for step in range(num_steps_per_epoch):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                observations = batch.observations.reshape(-1, self.argus.observation_dim)
                actions = batch.actions.reshape(-1, self.argus.action_dim)
                next_observations = batch.next_observations.reshape(-1, self.argus.observation_dim)
                rewards = batch.rewards.reshape(-1, 1)
                returns = batch.discounted_returns.reshape(-1, 1)
                dones = batch.dones.reshape(-1, 1)
                loss = {}
                if update_behavior:
                    energy_loss_info = self.behavior_flow_train(observations=observations, actions=actions)
                    loss.update(energy_loss_info)
                if update_flow:
                    flow_loss_info = self.flow_train(observations=observations, actions=actions)
                    loss.update(flow_loss_info)
                if update_flow_value:
                    flow_value_loss_info = self.dist_flow_value_train(
                        observations=observations, rewards=rewards, values=returns)
                    loss.update(flow_value_loss_info)
                if update_energy:
                    energy_loss_info = self.energy_train(
                        iql_tau=self.argus.iql_tau, observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(energy_loss_info)

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        value_test_info = self.value_test(observations=observations, actions=actions)
                        loss.update(value_test_info)
                    #     wandb_infos = {}
                    #     wandb_infos.update(loss)
                    #     # wandb_infos.update({"diffusion_loss": loss})
                    #     for info_key, info_val in wandb_infos.items():
                    #         wandb_infos[info_key] = info_val
                    #     wandb.log(wandb_infos, step=self.step)
                    wandb.log(loss, step=self.step)


                if (self.step+1) % 500 == 0 and update_flow:
                    self.behavior_eval()
                    self.eval()

                if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                    self.save_critic_checkpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def behavior_eval(self):
        print(f"Perform behavior flow evaluation on guidance_scale ......")
        eval_results = parallel_d4rl_eval_behavior_flow(
            argus=self.argus, dataset=self.dataset, model=self.behavior_flow,
            critic=None, guidance_scale=1.0, eval_episodes=self.argus.eval_episodes,
            ddim_sample=True)
        if self.wandb_log:
            wandb.log(eval_results, step=self.step)

    def eval(self):
        print(f"Perform evaluation on guidance_scale ......")
        eval_results = parallel_d4rl_eval_adaptive_flow_step(
            argus=self.argus, dataset=self.dataset, model=self.model,
            critic=None, guidance_scale=1.0, eval_episodes=self.argus.eval_episodes, ddim_sample=True)
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

    def save_critic_checkpoint(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {}
        if self.argus.critic_type in [CriticType.iql, CriticType.ciql]:
            data.update({
                'step': self.step,
                'behavior_flow': self.behavior_flow.state_dict(),
            })
        elif self.argus.critic_type in [CriticType.isql]:
            data.update({
                'step': self.step,
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
