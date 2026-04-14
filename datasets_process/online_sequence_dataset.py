import torch
import numpy as np
from path_process.get_path import get_project_path
from datasets_process.buffer import ReturnReplayBuffer, OnlineReturnReplayBuffer
from datasets_process.normalizer import DatasetNormalizer, OnlineDatasetNormalizer
from collections import namedtuple
from termcolor import colored
import time
import random
from datasets_process.dataset_util import load_environment, get_environment_info, d4rl_trajectories_iterator


Batch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns dones')
OnlineBatch = namedtuple('Batch', 'trajectories conditions observations actions action_log_probs rewards next_observations returns dones')
TaskCondBatch = namedtuple('TaskCondBatch', 'trajectories conditions task_identity observations actions rewards next_observations returns dones')
ContextBatch = namedtuple('ContextBatch', 'trajectories conditions observations actions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions observations actions next_observations rewards dones returns')

class OnlineSequenceDataset():
    def __init__(self, argus=None, project_path=None, env_name='ant_dir', domain="ant_dir", sequence_length=64,
        normalizer='OnlineGaussianNormalizer', termination_penalty=0, discount=0.99, returns_scale=1000):
        if project_path is None:
            project_path = get_project_path()
        self.project_path = project_path
        self.argus = argus
        self.domain = domain
        self.env_name = env_name
        self.env, self.observation_dim, self.action_dim, self.original_action_range = get_environment_info(env_name=env_name, domain=domain)
        self.eval_env, _, __, ___ = get_environment_info(env_name=env_name, domain=domain)
        self.transition_dim = self.observation_dim + self.action_dim
        self.max_path_length = self.env.max_episode_steps
        self.returns_scale = returns_scale
        self.history_max_normalized_return = 0
        self.history_observation_range = [-1.0, 1.0]
        self.history_max_action_range = 0
        self.sequence_length = sequence_length
        self.discount = discount
        self.termination_penalty = termination_penalty
        self.replay_buffer = self.get_data_from_dataset()
        self.normalizer = OnlineDatasetNormalizer(self.replay_buffer, normalizer)
        self.indices, self.total_indices, self.replay_buffer_ep_pointer = [], [], 0
        self.n_episodes = self.replay_buffer.n_episodes
        self.path_lengths = self.replay_buffer.path_lengths
        self.max_path_length = self.env.max_episode_steps
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.replay_buffer.finalize()
        print(self.replay_buffer)

    def get_data_from_dataset(self):
        replay_buffer = OnlineReturnReplayBuffer(
            argus=self.argus, termination_penalty=self.termination_penalty, discounts=self.discount, max_path_length=self.max_path_length,
            buffer_keys=["observations", "actions", "log_probs", "rewards", "next_observations", "dones", "terminals"],
            normalize_keys={"observations": self.observation_dim, "actions": self.action_dim, "next_observations": self.observation_dim}
        )
        replay_buffer.__setattr__("observation_dim", self.observation_dim)
        replay_buffer.__setattr__("action_dim", self.action_dim)
        return replay_buffer

    def store_trajectories(self, trajectories):
        assert isinstance(trajectories, list)
        for trajectory in trajectories:
            self.replay_buffer.add_path(path=trajectory)
            self.make_indices(self.replay_buffer._dict['path_lengths'][-1])
        if self.argus.preserve_ep > 0:
            if len(self.replay_buffer._dict['path_lengths']) > self.argus.preserve_ep:
                self._trim_replay_buffer_to_recent_episodes(self.argus.preserve_ep)

    def assign_normalizer_parameters(self):
        self.normalizer.calculate_normalize_parameters(self.replay_buffer)


    def get_max_min_discounted_return(self):
        return np.max(np.vstack(self.replay_buffer._dict['discounted_returns'])), np.min(np.vstack(self.replay_buffer._dict['discounted_returns']))

    def normalize(self, keys=['observations', 'actions', 'next_observations']):
        '''
            'fft_observations'
            normalize fields that will be predicted by the diffusion model
        '''
        if self.env_name.split("-")[0] in ["hammer", "pen", "relocate", "door"]:
            keys = ['observations', 'actions']
        for key in keys:
            self.replay_buffer[f'normed_{key}'] = []
            for path_i, path in enumerate(self.replay_buffer[key]):
                self.replay_buffer[f'normed_{key}'].append(self.normalizer(path, key))

    def make_indices(self, path_length):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        max_start = min(path_length, path_length - self.sequence_length)
        for start in range(max_start + 1):
            end = start + self.sequence_length
            self.indices.append((self.replay_buffer_ep_pointer, start, end))
            self.total_indices.append((self.replay_buffer_ep_pointer, start, end))
        self.replay_buffer_ep_pointer += 1

    def _rebuild_indices(self):
        self.indices = []
        self.total_indices = []
        self.replay_buffer_ep_pointer = 0
        for path_length in self.replay_buffer._dict['path_lengths']:
            self.make_indices(path_length)

    def _trim_replay_buffer_to_recent_episodes(self, preserve_ep):
        keep_slice = slice(-preserve_ep, None)
        for key, value in list(self.replay_buffer._dict.items()):
            if isinstance(value, list):
                self.replay_buffer._dict[key] = value[keep_slice]
        self.replay_buffer._count = len(self.replay_buffer._dict['path_lengths'])
        self.replay_buffer._add_attributes()
        self._rebuild_indices()

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        if len(np.shape(observations)) > 2:
            return {0: observations[:, 0, :]}
        else:
            return {0: observations[0]}

    def sample_trajectories(self, batch_size, indices_type):
        sample_idxs = random.sample(range(self.__len__(indices_type)), batch_size)
        observations, next_observations, actions, actions_log_probs, rewards, returns, dones = [], [], [], [], [], [], []
        for idx in sample_idxs:
            obs, next_obs, act, act_log_probs, rew, ret, don = self.__getitem__(idx, indices_type=indices_type)
            observations.append(np.expand_dims(obs, axis=0))
            next_observations.append(np.expand_dims(next_obs, axis=0))
            actions.append(np.expand_dims(act, axis=0))
            actions_log_probs.append(np.expand_dims(act_log_probs, axis=0))
            rewards.append(np.expand_dims(rew, axis=0))
            returns.append(np.expand_dims(ret, axis=0))
            dones.append(np.expand_dims(don, axis=0))
        observations = np.vstack(observations)
        if np.max(observations) > self.history_observation_range[1]:
            self.history_observation_range[1] = np.max(observations)
        if np.min(observations) < self.history_observation_range[0]:
            self.history_observation_range[0] = np.min(observations)
        next_observations = np.vstack(next_observations)
        actions = np.vstack(actions)
        actions_log_probs = np.vstack(actions_log_probs)
        rewards = np.vstack(rewards)
        returns = np.vstack(returns)
        if np.max(returns) > self.history_max_normalized_return:
            self.history_max_normalized_return = np.max(returns)
        dones = np.vstack(dones)
        trajectories = np.concatenate([observations, actions], axis=-1)
        conditions = self.get_conditions(observations)
        batch = OnlineBatch(trajectories, conditions, observations, actions, actions_log_probs, rewards, next_observations, returns, dones)
        return batch

    def __len__(self, indices_type="diffusion"):
        if indices_type == "diffusion":
            return len(self.total_indices)
        elif indices_type == "ac":
            return len(self.indices)
        else:
            raise Exception("indices_type not defined")

    def __getitem__(self, idx, eps=1e-4, indices_type="diffusion"):
        if indices_type == "diffusion":
            indices = self.total_indices
        elif indices_type == "ac":
            indices = self.indices
        else:
            raise Exception("indices_type not defined")
        path_ind, start, end = indices[idx]
        observations = self.replay_buffer.observations[path_ind][start:end]
        next_observations = self.replay_buffer.next_observations[path_ind][start:end]
        actions = self.replay_buffer.actions[path_ind][start:end]
        action_log_probs = self.replay_buffer.log_probs[path_ind][start:end]
        rewards = self.replay_buffer.rewards[path_ind][start:end] / self.argus.reward_scale
        returns = self.replay_buffer.discounted_returns[path_ind][start] / self.argus.returns_scale
        dones = self.replay_buffer.dones[path_ind][start:end]
        if self.argus.train_with_normed_data:
            observations = self.normalizer.normalize(observations, "observations")
            next_observations = self.normalizer.normalize(next_observations, "observations")
            # actions = self.normalizer.normalize(actions, "actions")
        return observations, next_observations, actions, action_log_probs, rewards, returns, dones

class DiffusionValueBasedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, argus=None, project_path=None, task_idx=4, env_name='ant_dir', sequence_length=64,
                 normalizer='GaussianNormalizer', termination_penalty=0, discount=0.99, returns_scale=1000):
        if project_path is None:
            project_path = get_project_path()
        self.project_path = project_path
        self.argus = argus
        self.env_name = env_name
        self.task_idx = task_idx
        self.env = load_environment(env_name=env_name)
        self.eval_env = load_environment(env_name=env_name)
        self.max_path_length = self.env.max_episode_steps
        self.returns_scale = returns_scale
        self.sequence_length = sequence_length
        self.discount = discount
        self.termination_penalty = termination_penalty
        self.replay_buffer = self.get_data_from_dataset()
        self.normalizer = DatasetNormalizer(self.replay_buffer, normalizer, path_lengths=self.replay_buffer['path_lengths'])
        self.indices = self.make_indices(self.replay_buffer.path_lengths)
        self.observation_dim = self.replay_buffer.observations[0].shape[-1]
        self.action_dim = self.replay_buffer.actions[0].shape[-1]
        self.transition_dim = self.observation_dim + self.action_dim
        self.n_episodes = self.replay_buffer.n_episodes
        self.path_lengths = self.replay_buffer.path_lengths
        self.max_path_length = np.max(self.replay_buffer.path_lengths)
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.normalize()
        self.action_ranges = self.get_task_specific_action_ranges()
        print(self.replay_buffer)

    def get_task_specific_action_ranges(self):
        if self.argus.train_with_normed_data:
            action_ranges = (np.min(np.vstack(self.replay_buffer.normed_actions)), np.max(np.vstack(self.replay_buffer.normed_actions)))
        else:
            action_ranges = (-0.99999, 0.99999)
        return action_ranges

    def unnormalize_return(self, x):
        return x * (self.replay_buffer.max_return - self.replay_buffer.min_return + 1e-6) + self.replay_buffer.min_return

    def normalize_return(self, x):
        return (x - self.replay_buffer.min_return) / (self.replay_buffer.max_return - self.replay_buffer.min_return + 1e-6)

    def get_data_from_dataset(self):
        trajectories_iter = d4rl_trajectories_iterator(env=self.env)
        replay_buffer = ReturnReplayBuffer(argus=self.argus, termination_penalty=self.termination_penalty, discounts=self.discount, max_path_length=self.max_path_length)
        for i, episode in enumerate(trajectories_iter):
            episode_data, min_reward = episode
            if len(episode_data['observations']) <= 1000:
                replay_buffer.add_path(episode_data)
            if self.argus.partial_dataset_training:
                if i > self.argus.dataset_traj_num:
                    break
            if self.argus.debug_mode:
                if i > 100:
                    break
        replay_buffer.finalize()
        return replay_buffer

    def get_max_min_discounted_return(self):
        return np.max(np.vstack(self.replay_buffer._dict['discounted_returns'])), np.min(
            np.vstack(self.replay_buffer._dict['discounted_returns']))

    def normalize(self, keys=['observations', 'actions', 'next_observations']):
        '''
            'fft_observations'
            normalize fields that will be predicted by the diffusion model
        '''
        if self.env_name.split("-")[0] in ["hammer", "pen", "relocate", "door"]:
            keys = ['observations', 'actions']
        for key in keys:
            self.replay_buffer[f'normed_{key}'] = []
            for path_i, path in enumerate(self.replay_buffer[key]):
                self.replay_buffer[f'normed_{key}'].append(self.normalizer(path, key))

    def make_indices(self, path_lengths):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length, path_length - self.sequence_length)
            for start in range(max_start+1):
                end = start + self.sequence_length
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_Q_conditions(self, obs_act):
        return {"Q_conditions": obs_act}

    def sample(self, batch_size):
        data_info = np.random.choice(self.__len__(), size=batch_size, replace=False)
        observations, next_observations, actions, returns, rewards, dones = [], [], [], [], [], []
        for data_i in data_info:
            path_ind, start, end = self.indices[data_i]
            if self.argus.train_with_normed_data:
                observations.append(self.replay_buffer.normed_observations[path_ind][start:end])
                next_observations.append(self.replay_buffer.normed_next_observations[path_ind][start:end])
                actions.append(self.replay_buffer.normed_actions[path_ind][start:end])
            else:
                observations.append(self.replay_buffer.observations[path_ind][start:end])
                next_observations.append(self.replay_buffer.next_observations[path_ind][start:end])
                actions.append(self.replay_buffer.actions[path_ind][start:end])
            returns.append(self.replay_buffer.discounted_returns[path_ind][start:end])
            rewards.append(self.replay_buffer.rewards[path_ind][start:end])
            dones.append(self.replay_buffer.dones[path_ind][start:end])
        observations = torch.tensor(np.vstack(observations))
        actions = torch.tensor(np.vstack(actions))
        trajectories = torch.cat([observations, actions], dim=-1)
        conditions = self.get_Q_conditions(trajectories[:, 0, :])
        batch = ValueBatch(trajectories, conditions, observations, actions, next_observations, rewards, dones, returns)
        return batch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        if self.argus.train_with_normed_data:
            observations = self.replay_buffer.normed_observations[path_ind][start:end]
            next_observations = self.replay_buffer.normed_next_observations[path_ind][start:end]
            actions = self.replay_buffer.normed_actions[path_ind][start:end]
        else:
            observations = self.replay_buffer.observations[path_ind][start:end]
            next_observations = self.replay_buffer.next_observations[path_ind][start:end]
            actions = self.replay_buffer.actions[path_ind][start:end]
        trajectories = np.concatenate([observations, actions], axis=-1)
        conditions = self.get_Q_conditions(trajectories[0, :])
        returns = self.replay_buffer.discounted_returns[path_ind][start:end]
        rewards = self.replay_buffer.rewards[path_ind][start:end]
        dones = self.replay_buffer.terminals[path_ind][start:end]
        batch = ValueBatch(trajectories, conditions, observations, actions, next_observations, rewards, dones, returns)
        return batch


class DiffusionDenoiseSequenceDataset(DiffusionValueBasedSequenceDataset):

    def make_indices(self, path_lengths):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length, path_length - self.sequence_length)
            min_start = max(0, self.argus.action_context_length)
            for start in range(min_start, max_start+1):
                end = start + self.sequence_length
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    def get_action_context(self, actions, rewards, path_ind, start):
        if start - self.argus.action_context_length >= 0:
            conditions = np.concatenate([
                actions[path_ind][start - self.argus.action_context_length:start],
                rewards[path_ind][start - self.argus.action_context_length:start],
            ], axis=-1)
        else:
            padding_ones = np.ones((self.argus.action_context_length-start, self.argus.act_embed_dim + 1))
            if start != 0:
                conditions = np.concatenate([
                    padding_ones,
                    np.concatenate([actions[path_ind][0:start],rewards[path_ind][0:start]], axis=-1),
                ], axis=0)
            else:
                conditions = padding_ones
        conditions = np.reshape(conditions, [self.argus.action_context_length*(self.argus.act_embed_dim+1),])
        return conditions

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        if self.argus.train_with_normed_data:
            observations = self.replay_buffer.normed_observations[path_ind][start:end]
            next_observations = self.replay_buffer.normed_next_observations[path_ind][start:end]
            actions = self.replay_buffer.normed_actions[path_ind][start:end]
            conditions = self.get_action_context(self.replay_buffer.normed_actions, self.replay_buffer.rewards, path_ind, start)
        else:
            observations = self.replay_buffer.observations[path_ind][start:end]
            next_observations = self.replay_buffer.next_observations[path_ind][start:end]
            actions = self.replay_buffer.actions[path_ind][start:end]
            conditions = self.get_action_context(self.replay_buffer.actions, self.replay_buffer.rewards, path_ind, start)
        trajectories = observations
        returns = self.replay_buffer.discounted_returns[path_ind][start]
        rewards = self.replay_buffer.rewards[path_ind][start:end]
        dones = self.replay_buffer.terminals[path_ind][start:end]
        batch = ValueBatch(trajectories, conditions, observations, actions, next_observations, rewards, dones, returns)
        return batch

if __name__ == "__main__":
    dataset = OnlineSequenceDataset(env_name="ant_dir")
