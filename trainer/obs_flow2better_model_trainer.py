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
from evaluation.sequential_eval import parallel_d4rl_eval_score_function_version
from models.flow_model import InverseDynamicsFlowMatchingNet

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
    def __init__(self, argus, model, inverse_dynamic, energy_model, dataset):
        self.argus = argus
        self.model = model
        self.model.to(self.argus.device)
        self.inverse_dynamic = inverse_dynamic
        self.inverse_dynamic.to(self.argus.device)
        self.energy_model = energy_model
        self.energy_model.to(self.argus.device)
        self.invdyn_flow = InverseDynamicsFlowMatchingNet(argus=argus, flow=self.model, inverse_dynamic=self.inverse_dynamic)
        self.dataset = dataset
        self.device = argus.device
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=argus.lr)
        self.invdyn_optimizer = torch.optim.Adam(self.inverse_dynamic.parameters(), lr=argus.lr)
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
        self.best_model_info = {"guidance_scale": 0.0, "performance": 0.0, "performance_std": 0.0, "seed": 0}
        self.dataloader = cycle_dataloader(argus=self.argus, dataset=self.dataset, train_batch_size=argus.batch_size)
        if self.wandb_log:
            wandb.init(name=self.wandb_exp_name, group=self.wandb_exp_group, project=self.wandb_project_name, config=obj2dict(self.argus))

    def get_detailed_save_path(self):
        return os.path.join(
            self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.current_exp_label)

    def observation_flow_train(self, observations, next_observations):
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=None, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, action_imitation_model=None,
            weighted_samples_type=WeightedSamplesType.specific_obs_x0x1,
            next_observations=next_observations, x_0=next_observations,
        )
        x_t_, t_, dx_dt_, weights_ = sample_weighted_interpolated_points(
            argus=self.argus, observations=next_observations, actions=None, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, action_imitation_model=None,
            weighted_samples_type=WeightedSamplesType.specific_obs_x0x1,
            next_observations=next_observations, x_0=next_observations,
        )
        if self.argus.time_rescale:
            t_ = t_ / 2.0
            t = t / 2.0 + 0.5
        x_t = torch.cat((x_t, x_t_), dim=0)
        t = torch.cat((t, t_), dim=0)
        dx_dt = torch.cat((dx_dt, dx_dt_), dim=0)
        weights = torch.cat((weights, weights_), dim=0)
        v_pred = self.model(x_t, t)
        loss = torch.mean((v_pred - dx_dt)**2 * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"flow_loss": loss.detach().cpu().numpy().item()}

    def inverse_dynamic_train(self, observations, next_observations, actions):
        a_pred = self.inverse_dynamic(torch.cat([observations, next_observations], dim=-1))
        loss = self.loss_fn(a_pred, actions)
        self.invdyn_optimizer.zero_grad()
        loss.backward()
        self.invdyn_optimizer.step()
        return {"invdyn_loss": loss.detach().cpu().numpy().item()}

    def flow_train(self, observations, next_observations, actions):
        if self.argus.flow_guided_mode == FlowGuidedMode.normal:
            loss = {}
            flow_loss_info = self.observation_flow_train(
                observations=observations, next_observations=next_observations)
            loss.update(flow_loss_info)
            invdyn_loss_info = self.inverse_dynamic_train(
                observations=observations, next_observations=next_observations, actions=actions
            )
            loss.update(invdyn_loss_info)
            return loss
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
                    flow_loss_info = self.flow_train(
                        observations=observations, next_observations=next_observations, actions=actions)
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
        eval_results = parallel_d4rl_eval_score_function_version(
            argus=self.argus, dataset=self.dataset, model=self.invdyn_flow,
            critic=self.energy_model, guidance_scale=1.0,
            eval_episodes=self.argus.eval_episodes, ddim_sample=True)
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
