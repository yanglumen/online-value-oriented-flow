import os.path
import wandb
import torch
import random
import torch.nn as nn
import numpy as np
from models.f2b_forward_process import sample_weighted_interpolated_points
from trainer.trainer_util import (
    batch_to_device,
)
from termcolor import colored
from config.dict2class import obj2dict
from config.f2b_hyperparameter import *
import matplotlib.pyplot as plt
from evaluation.sequential_eval import flow2better_d4rl_eval

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
    def __init__(self, argus, model, action_imitation_model, energy_model, inverse_dynamics_model, dataset):
        self.argus = argus
        self.model = model
        self.model.to(self.argus.device)
        self.action_imitation_model = action_imitation_model
        self.action_imitation_model.to(self.argus.device)
        self.energy_model = energy_model
        self.energy_model.to(self.argus.device)
        self.inv_model = inverse_dynamics_model
        self.inv_model.to(self.argus.device)
        self.dataset = dataset
        self.device = argus.device
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=argus.lr)
        self.act_imt_optimizer = torch.optim.Adam(self.action_imitation_model.parameters(), lr=argus.lr)
        self.energy_q_optimizer = torch.optim.Adam(self.energy_model.q.parameters(), lr=argus.lr)
        self.energy_v_optimizer = torch.optim.Adam(self.energy_model.v.parameters(), lr=argus.lr)
        self.inv_optimizer = torch.optim.Adam(self.inv_model.parameters(), lr=argus.lr)
        self.step = 0
        self.save_path = argus.save_path
        self.save_freq = argus. save_freq
        self.env_name = argus.dataset
        self.wandb_log = argus.wandb_log
        self.wandb_exp_name = argus.wandb_exp_name
        self.wandb_exp_group = argus.wandb_exp_group
        self.wandb_log_frequency = argus.wandb_log_frequency
        self.wandb_project_name = argus.wandb_project_name
        self.best_model_info = {"guidance_scale": 0.0, "performance": 0.0, "performance_std": 0.0, "seed": 0}
        self.dataloader = cycle_dataloader(argus=self.argus, dataset=self.dataset, train_batch_size=argus.batch_size)
        if self.wandb_log:
            wandb.init(name=self.wandb_exp_name, group=self.wandb_exp_group, project=self.wandb_project_name, config=obj2dict(self.argus))

    def get_detailed_save_path(self):
        return os.path.join(
            self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.current_exp_label)

    def action_imitation_flow_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, action_imitation_model=None,
            weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        v_pred = self.action_imitation_model(x_t, t)
        loss = self.loss_fn(v_pred, dx_dt)
        self.act_imt_optimizer.zero_grad()
        loss.backward()
        self.act_imt_optimizer.step()
        return {"act_imt_flow_loss": loss.detach().cpu().numpy().item()}

    def guided_flow_train(self, observations, actions, next_observations, discounted_returns, next_discounted_returns):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions,
            energy=discounted_returns, next_energy=next_discounted_returns, beta=self.argus.beta,
            energy_model=self.energy_model, action_imitation_model=self.action_imitation_model,
            next_observations=next_observations, weighted_samples_type=WeightedSamplesType.subopt_state2opt_state,
        )
        # energy = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        # unweighted_x_t, unweighted_t, unweighted_dx_dt, unweighted_weights = sample_weighted_interpolated_points(
        #     argus=self.argus, observations=observations, actions=actions, energy=energy, beta=self.argus.beta,
        #     energy_model=self.energy_model, action_imitation_model=self.action_imitation_model,
        #     weighted_samples_type=WeightedSamplesType.subopt2opt,
        # )
        # weighted_x_t, weighted_t, weighted_dx_dt, weighted_weights = sample_weighted_interpolated_points(
        #     argus=self.argus, observations=observations, actions=actions, energy=energy, beta=self.argus.beta,
        #     energy_model=self.energy_model, action_imitation_model=self.action_imitation_model,
        #     weighted_samples_type=WeightedSamplesType.noise_interpolation,
        # )
        # if self.argus.time_rescale:
        #     unweighted_t = unweighted_t / 2.0
        #     weighted_t = weighted_t / 2.0 + 0.5
        # x_t = torch.cat((unweighted_x_t, weighted_x_t), dim=0)
        # t = torch.cat((unweighted_t, weighted_t), dim=0)
        # dx_dt = torch.cat((unweighted_dx_dt, weighted_dx_dt), dim=0)
        # weights = torch.cat((unweighted_weights, weighted_weights), dim=0)
        # # weights = weights / weights.sum()
        v_pred = self.model(x_t, t)
        # with torch.no_grad():
        #     imitation_pred = self.action_imitation_model(x_t, t)
        # loss = torch.mean((v_pred - dx_dt)**2 * weights) + self.loss_fn(v_pred, imitation_pred.detach())
        loss = torch.mean((v_pred - dx_dt)**2 * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"flow_loss": loss.detach().cpu().numpy().item()}

    def gradient_guided_flow_train(self, observations, actions):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
        )
        with torch.enable_grad():
            x_t.requires_grad_(True)
            pred_e = self.energy_model.get_scaled_q(
                obs=observations, act=x_t[:, self.argus.observation_dim:], scale=self.argus.energy_scale)
            grad_of_x_t = torch.autograd.grad(torch.sum(pred_e), x_t)[0][:, self.argus.observation_dim:]
        grad_of_x_t = grad_of_x_t / torch.norm(grad_of_x_t, dim=-1, keepdim=True) * torch.norm(dx_dt, dim=-1, keepdim=True)
        v_pred = self.model(x_t, t)
        loss = self.loss_fn(v_pred, grad_of_x_t.detach())# + 0.1*torch.mean((v_pred - dx_dt.detach()) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"flow_loss": loss.detach().cpu().numpy().item()}

    def flow_train(self, observations, actions, next_observations, discounted_returns, next_discounted_returns):
        if self.argus.flow_guided_mode == FlowGuidedMode.normal:
            flow_loss_info = self.guided_flow_train(
                observations=observations, actions=actions, next_observations=next_observations,
                discounted_returns=discounted_returns, next_discounted_returns=next_discounted_returns)
            return flow_loss_info
        if self.argus.flow_guided_mode == FlowGuidedMode.gradient:
            flow_loss_info = self.gradient_guided_flow_train(observations=observations, actions=actions)
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

    def inverse_dynamics_train(self, observations, actions, next_observations):
        s_next_s = torch.cat((observations, next_observations), dim=-1)
        pred_a = self.inv_model(state=s_next_s)
        loss = self.loss_fn(pred_a, actions)
        self.inv_optimizer.zero_grad()
        loss.backward()
        self.inv_optimizer.step()
        return {"inverse_dynamic_loss": loss.detach().cpu().numpy().item()}

    def guided_train(self, num_epochs, num_steps_per_epoch):
        try:
            self.load_critic_checkpoint(os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.load_critic_model))
            epoch_offset = 40
        except:
            epoch_offset = 0
        for epoch in range(epoch_offset, num_epochs):
            for step in range(num_steps_per_epoch):
                if self.argus.debug_mode:
                    update_energy = True
                    update_behavior = True
                    update_flow = True
                    update_inv = True
                else:
                    update_energy = True if epoch < 40 else False
                    update_behavior = True if epoch < 40 else False
                    update_flow = True if epoch < 40 else True
                    update_inv = True if epoch < 40 else True
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                observations = batch.observations.reshape(-1, self.argus.observation_dim)
                actions = batch.actions.reshape(-1, self.argus.action_dim)
                next_observations = batch.next_observations.reshape(-1, self.argus.observation_dim)
                rewards = batch.rewards.reshape(-1, 1)
                discounted_returns = batch.discounted_returns.reshape(-1, 1)
                next_discounted_returns = batch.next_discounted_returns.reshape(-1, 1)
                dones = batch.dones.reshape(-1, 1)
                loss = {}
                if update_energy:
                    energy_loss_info = self.energy_train(
                        iql_tau=self.argus.iql_tau, observations=observations, actions=actions,
                        next_observations=next_observations, rewards=rewards, dones=dones)
                    loss.update(energy_loss_info)
                if update_behavior:
                    flow_loss_info = self.action_imitation_flow_train(observations=observations, actions=actions)
                    loss.update(flow_loss_info)
                if update_flow:
                    flow_loss_info = self.flow_train(
                        observations=observations, actions=actions, next_observations=next_observations,
                        discounted_returns=discounted_returns, next_discounted_returns=next_discounted_returns)
                    loss.update(flow_loss_info)
                if update_inv:
                    inv_loss_info = self.inverse_dynamics_train(
                        observations=observations, actions=actions, next_observations=next_observations)
                    loss.update(inv_loss_info)

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        wandb.log(loss, step=self.step)
                        # wandb_infos = {}
                        # wandb_infos.update(loss)xx
                        # # wandb_infos.update({"diffusion_loss": loss})
                        # for info_key, info_val in wandb_infos.items():
                        #     wandb_infos[info_key] = info_val
                        # wandb.log(wandb_infos, step=self.step)

                if self.step % 5000 == 0 and update_flow:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def iterate_train(self, num_epochs, num_steps_per_epoch):
        self.load_iterate_checkpoint(
            best_flow_loadpath=os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), "ciql_f2b_iterate"),
            critic_loadpath=os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), "ciql_f2b")
        )
        epoch_offset = 40
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
                    flow_loss_info = self.action_imitation_flow_train(observations=observations, actions=actions)
                    loss.update(flow_loss_info)
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

                if (self.step+1) % 5000 == 0 and update_flow:
                    self.eval()

                if update_energy:
                    if self.step % self.save_freq == 0 or (self.step % (num_epochs * num_steps_per_epoch) == (num_epochs * num_steps_per_epoch - 1)):
                        self.save_critic_checpoint()

                if self.step % 1000 == 0:
                    print(f"Epoch {epoch} | BestScore {self.best_model_info['performance']} | Step {self.step} | Loss: {loss}")
                self.step += 1

    def evaluation(self):
        self.load_best_flow_checkpoint(
            os.path.join(self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")),
                         self.argus.load_critic_model))
        self.eval()
        print(colored(f'Evaluation Done', color="cyan"))

    def eval(self):
        print(f"Perform evaluation on guidance_scale ......")
        self.model.eval()
        eval_results = flow2better_d4rl_eval(
            argus=self.argus, dataset=self.dataset, model=self.model, inverse_dynamics=self.inv_model,
            critic=self.energy_model, guidance_scale=1.0, eval_episodes=self.argus.eval_episodes, ddim_sample=True)
        for key, val in eval_results.items():
            if "ave_score" in key:
                if self.best_model_info["performance"] < val:
                    self.best_model_info["performance"] = val
                    self.best_model_info["seed"] = self.argus.seed
                    for key__, val__ in eval_results.items():
                        if "std_score" in key__:
                            self.best_model_info["performance_std"] = val__
                    data = {
                        'guidance_scale': self.best_model_info["guidance_scale"],
                        'performance': self.best_model_info["performance"],
                        'seed': self.best_model_info["seed"],
                        'step': self.step,
                        'flow': self.model.state_dict(),
                    }
                    savepath = os.path.join(self.get_detailed_save_path(), 'best_flow_checkpoint')
                    os.makedirs(savepath, exist_ok=True)
                    torch.save(data, os.path.join(savepath, f'flow_{self.step}.pt'))
                    print(colored(f'[ utils/training ] Saved best model to {savepath}', color="green"))
        if self.wandb_log:
            wandb.log(eval_results, step=self.step)
        self.model.train()

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
                'imt_act': self.action_imitation_model.state_dict(),
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
            self.action_imitation_model.load_state_dict(data['imt_act'])
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

    def load_iterate_critic_checkpoint(self, loadpath, step_offset="latest"):
        step = self.get_checkpoint_step(
            loadpath=os.path.join(loadpath, 'critic_checkpoint'), step_offset=step_offset)
        loadpath = os.path.join(loadpath, f'critic_checkpoint/critic_{step}.pt')
        data = torch.load(loadpath)
        print(colored(f" From {loadpath} Load Critic Model with saved train step {step} Success !!!", color="green"))
        if self.argus.critic_type in [CriticType.iql, CriticType.ciql]:
            self.energy_model.q.load_state_dict(data['q'])
            self.energy_model.q_target.load_state_dict(data['q_target'])
            self.energy_model.v.load_state_dict(data['v'])
            self.action_imitation_model.load_state_dict(self.model.state_dict())
        elif self.argus.critic_type in CriticType.cql:
            pass
        else:
            raise ValueError(f"Critic type {self.argus.critic_type} not supported")

    def load_iterate_checkpoint(self, best_flow_loadpath, critic_loadpath, step_offset="latest"):
        self.load_best_flow_checkpoint(best_flow_loadpath, step_offset)
        self.load_iterate_critic_checkpoint(critic_loadpath, step_offset)
