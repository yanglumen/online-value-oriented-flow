import torch
import numpy as np
from path_process.get_path import get_project_path
from datasets_process.buffer import ReturnReplayBuffer, FlowTFReplayBuffer
from datasets_process.normalizer import DatasetNormalizer
from collections import namedtuple
from termcolor import colored
import math
import time
from datasets_process.dataset_util import load_environment, d4rl_trajectories_iterator


Batch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns dones')
CSG_Batch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns dones etas etas_idx')
IDBatch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns dones fake_actions fake_next_actions')
DFVBatch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns discounted_returns dones fake_actions fake_next_actions')
F2BBatch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns discounted_returns next_discounted_returns dones fake_actions fake_next_actions')
TaskCondBatch = namedtuple('TaskCondBatch', 'trajectories conditions task_identity observations actions rewards next_observations returns dones')
ContextBatch = namedtuple('ContextBatch', 'trajectories conditions observations actions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions observations actions next_observations rewards dones returns')
FlowTFBatch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns ep_returns dones')

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, argus=None, project_path=None, env_name='ant_dir', domain='mujoco', sequence_length=64,
        normalizer='GaussianNormalizer', termination_penalty=0, discount=0.99, returns_scale=1000):
        if project_path is None:
            project_path = get_project_path()
        self.project_path = project_path
        self.argus = argus
        self.env_name = env_name
        self.domain = domain
        self.env = load_environment(env_name=env_name, domain=domain)
        self.eval_env = load_environment(env_name=env_name, domain=domain)
        self.eval_envs = [load_environment(env_name=self.env_name, domain=argus.domain) for _ in range(argus.eval_episodes)]
        if 'maze2d' in self.argus.dataset:
            self.max_path_length = self.env.max_episode_steps #30500
        elif 'antmaze' in self.argus.dataset:
            self.max_path_length = 3500
        else:
            self.max_path_length = self.env.max_episode_steps
        self.max_action_val = self.env.action_space.high[0]
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
        self.data_range = self.get_data_range()
        self.normalize()
        print(self.replay_buffer)

    def get_data_range(self):
        data_range = {
            "observations_range": [np.min((np.min(self.normalizer.normalizers["observations"].mins), self.argus.eta - 3)),
                                   np.max((np.max(self.normalizer.normalizers["observations"].maxs), self.argus.eta + 3))],
            "normed_observations_range": [self.argus.eta - 3, self.argus.eta + 3],
            # "next_observations_range": [np.min(self.normalizer.normalizers["next_observations"].mins), np.max(self.normalizer.normalizers["next_observations"].maxs)],
            # "normed_next_observations_range": [-self.argus.eta, self.argus.eta],
            "actions_range": [np.min((np.min(self.normalizer.normalizers["actions"].mins), self.argus.eta - 3)),
                              np.max((np.max(self.normalizer.normalizers["actions"].maxs), self.argus.eta + 3))],
            "normed_actions_range": [self.argus.eta - 3, self.argus.eta + 3],
        }
        return data_range

    def get_data_from_dataset(self):
        trajectories_iter = d4rl_trajectories_iterator(
            env=self.env, reward_tune=self.argus.reward_tune, CEP_dataset_load_mode=self.argus.CEP_dataset_load_mode, rl_mode=self.argus.rl_mode)
        replay_buffer = ReturnReplayBuffer(argus=self.argus, termination_penalty=self.termination_penalty, discounts=self.discount, max_path_length=self.max_path_length)
        for i, episode in enumerate(trajectories_iter):
            episode_data, min_reward = episode
            # if self.argus.dataset == "walker2d-medium-replay-v2":
            #     if len(episode_data['terminals']) >= 100:
            #         replay_buffer.add_path(episode_data)
            # else:
            if 'maze2d' in str(self.argus.dataset).lower():
                if len(episode_data['terminals']) >= 2:
                    replay_buffer.add_path(episode_data)
            else:
                replay_buffer.add_path(episode_data)
            if self.argus.debug_mode:
                if i > 100:
                    print(colored("This is debug mode !!!  This is debug mode !!!   This is debug mode !!!   This is debug mode !!!", "red"))
                    print(colored("This is debug mode !!!  This is debug mode !!!   This is debug mode !!!   This is debug mode !!!", "red"))
                    print(colored("This is debug mode !!!  This is debug mode !!!   This is debug mode !!!   This is debug mode !!!", "red"))
                    break
        replay_buffer.finalize()
        return replay_buffer

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
                if key == "next_observations":
                    normed_val = self.normalizer(path, "observations")
                else:
                    normed_val = self.normalizer(path, key)
                    self.data_range[f'normed_{key}_range'] = [
                        np.min((np.min(normed_val), self.data_range[f'normed_{key}_range'][0])),
                        np.max((np.max(normed_val), self.data_range[f'normed_{key}_range'][1]))]
                self.replay_buffer[f'normed_{key}'].append(normed_val)

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

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        if self.argus.train_with_normed_data:
            observations = self.replay_buffer.normed_observations[path_ind][start:end]
            next_observations = self.replay_buffer.normed_next_observations[path_ind][start:end]
            # actions = self.replay_buffer.normed_actions[path_ind][start:end]
        else:
            observations = self.replay_buffer.observations[path_ind][start:end]
            next_observations = self.replay_buffer.next_observations[path_ind][start:end]
        actions = self.replay_buffer.actions[path_ind][start:end]
        trajectories = np.concatenate([observations, actions], axis=-1)
        conditions = self.get_conditions(observations)
        rewards = self.replay_buffer.rewards[path_ind][start:end]
        returns = self.replay_buffer.discounted_returns[path_ind][start]
        dones = self.replay_buffer.terminals[path_ind][start:end]
        if len(self.argus.multi_etas) != 0:
            etas, etas_idx = self.replay_buffer.check_returns_separation(float(returns))
            etas = np.array([etas])
            etas_idx = np.array([etas_idx])
        else:
            etas = np.array([0])
            etas_idx = np.array([0])
        batch = CSG_Batch(trajectories, conditions, observations, actions, rewards, next_observations, returns, dones, etas, etas_idx)
        return batch

class InDistributionSequenceDataset(SequenceDataset):

    def check(self):
        for idx in range(len(self.indices)):
            path_ind, start, end = self.indices[idx]
            if end >= self.replay_buffer.path_lengths[path_ind]:
                raise Exception(f"path_ind:{path_ind}, start:{start}, end:{end}")

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        if self.argus.train_with_normed_data:
            observations = self.replay_buffer.normed_observations[path_ind][start:end]
            next_observations = self.replay_buffer.normed_next_observations[path_ind][start:end]
            # actions = self.replay_buffer.normed_actions[path_ind][start:end]
        else:
            observations = self.replay_buffer.observations[path_ind][start:end]
            next_observations = self.replay_buffer.next_observations[path_ind][start:end]
        actions = self.replay_buffer.actions[path_ind][start:end]
        trajectories = np.concatenate([observations, actions], axis=-1)
        conditions = self.get_conditions(observations)
        rewards = self.replay_buffer.rewards[path_ind][start:end]
        returns = self.replay_buffer.discounted_returns[path_ind][start]
        discounted_returns = self.replay_buffer.discounted_returns[path_ind][start:end]
        dones = self.replay_buffer.terminals[path_ind][start:end]
        try:
            # random_indices = torch.randint(0, self.argus.fake_action_per_state, (1,))
            fake_actions = self.replay_buffer.fake_actions[path_ind][start:end]
            fake_next_actions = self.replay_buffer.fake_next_actions[path_ind][start:end]
        except:
            fake_actions = 0
            fake_next_actions = 0
        batch = DFVBatch(
            trajectories, conditions, observations, actions, rewards, next_observations, returns, discounted_returns, dones, fake_actions, fake_next_actions)
        return batch

    def state_cluster_according_to_value_range(self, cluster_granularity=1):
        value_info = {}
        total_value_info = []
        cluster_num = math.ceil((self.replay_buffer.max_return - self.replay_buffer.min_return)/cluster_granularity)
        for i in range(cluster_num):
            total_value_info.append(0)
            low = self.replay_buffer.min_return + i * cluster_granularity
            high = self.replay_buffer.min_return + (i+1) * cluster_granularity
            value_info.update({f"range_[{low}-{high}]": 0})
        for ep_i in self.replay_buffer.discounted_returns:
            for r_i in ep_i:
                idx = math.floor((r_i - self.replay_buffer.min_return)/cluster_granularity)
                low = self.replay_buffer.min_return + idx * cluster_granularity
                high = self.replay_buffer.min_return + (idx + 1) * cluster_granularity
                value_info[f"range_[{low}-{high}]"] += 1
        print(value_info)

class Flow2BetterSequenceDataset(SequenceDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check(self):
        for idx in range(len(self.indices)):
            path_ind, start, end = self.indices[idx]
            if end >= self.replay_buffer.path_lengths[path_ind]:
                raise Exception(f"path_ind:{path_ind}, start:{start}, end:{end}")

    def make_indices(self, path_lengths):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length, path_length - self.sequence_length)
            for start in range(max_start):
                end = start + self.sequence_length
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        if self.argus.train_with_normed_data:
            observations = self.replay_buffer.normed_observations[path_ind][start:end]
            next_observations = self.replay_buffer.normed_next_observations[path_ind][start:end]
            # actions = self.replay_buffer.normed_actions[path_ind][start:end]
        else:
            observations = self.replay_buffer.observations[path_ind][start:end]
            next_observations = self.replay_buffer.next_observations[path_ind][start:end]
        actions = self.replay_buffer.actions[path_ind][start:end]
        trajectories = np.concatenate([observations, actions], axis=-1)
        conditions = self.get_conditions(observations)
        rewards = self.replay_buffer.rewards[path_ind][start:end]
        returns = self.replay_buffer.discounted_returns[path_ind][start]
        discounted_returns = self.replay_buffer.discounted_returns[path_ind][start:end]
        next_discounted_returns = self.replay_buffer.discounted_returns[path_ind][start+1:end+1]
        dones = self.replay_buffer.terminals[path_ind][start:end]
        try:
            # random_indices = torch.randint(0, self.argus.fake_action_per_state, (1,))
            fake_actions = self.replay_buffer.fake_actions[path_ind][start:end]
            fake_next_actions = self.replay_buffer.fake_next_actions[path_ind][start:end]
        except:
            fake_actions = 0
            fake_next_actions = 0
        batch = F2BBatch(
            trajectories, conditions, observations, actions, rewards, next_observations, returns, discounted_returns, next_discounted_returns, dones, fake_actions, fake_next_actions)
        return batch

    def state_cluster_according_to_value_range(self, cluster_granularity=1):
        value_info = {}
        total_value_info = []
        cluster_num = math.ceil((self.replay_buffer.max_return - self.replay_buffer.min_return)/cluster_granularity)
        for i in range(cluster_num):
            total_value_info.append(0)
            low = self.replay_buffer.min_return + i * cluster_granularity
            high = self.replay_buffer.min_return + (i+1) * cluster_granularity
            value_info.update({f"range_[{low}-{high}]": 0})
        for ep_i in self.replay_buffer.discounted_returns:
            for r_i in ep_i:
                idx = math.floor((r_i - self.replay_buffer.min_return)/cluster_granularity)
                low = self.replay_buffer.min_return + idx * cluster_granularity
                high = self.replay_buffer.min_return + (idx + 1) * cluster_granularity
                value_info[f"range_[{low}-{high}]"] += 1
        print(value_info)

class FlowTFSequenceDataset(SequenceDataset):

    def get_data_from_dataset(self):
        trajectories_iter = d4rl_trajectories_iterator(
            env=self.env, reward_tune=self.argus.reward_tune, CEP_dataset_load_mode=self.argus.CEP_dataset_load_mode)
        replay_buffer = FlowTFReplayBuffer(argus=self.argus, termination_penalty=self.termination_penalty, discounts=self.discount, max_path_length=self.max_path_length)
        for i, episode in enumerate(trajectories_iter):
            episode_data, min_reward = episode
            # if self.argus.dataset == "walker2d-medium-replay-v2":
            #     if len(episode_data['terminals']) >= 100:
            #         replay_buffer.add_path(episode_data)
            # else:
            if 'maze2d' in str(self.argus.dataset).lower():
                if len(episode_data['terminals']) >= 2:
                    replay_buffer.add_path(episode_data)
            else:
                replay_buffer.add_path(episode_data)
            if self.argus.debug_mode:
                if i > 100:
                    print(colored("This is debug mode !!!  This is debug mode !!!   This is debug mode !!!   This is debug mode !!!", "red"))
                    print(colored("This is debug mode !!!  This is debug mode !!!   This is debug mode !!!   This is debug mode !!!", "red"))
                    print(colored("This is debug mode !!!  This is debug mode !!!   This is debug mode !!!   This is debug mode !!!", "red"))
                    break
        replay_buffer.finalize()
        return replay_buffer

    def check(self):
        for idx in range(len(self.indices)):
            path_ind, start, end = self.indices[idx]
            if end >= self.replay_buffer.path_lengths[path_ind]:
                raise Exception(f"path_ind:{path_ind}, start:{start}, end:{end}")

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        if self.argus.train_with_normed_data:
            observations = self.replay_buffer.normed_observations[path_ind][start:end]
            next_observations = self.replay_buffer.normed_next_observations[path_ind][start:end]
            # actions = self.replay_buffer.normed_actions[path_ind][start:end]
        else:
            observations = self.replay_buffer.observations[path_ind][start:end]
            next_observations = self.replay_buffer.next_observations[path_ind][start:end]
        actions = self.replay_buffer.actions[path_ind][start:end]
        trajectories = np.concatenate([observations, actions], axis=-1)
        conditions = self.get_conditions(observations)
        rewards = self.replay_buffer.rewards[path_ind][start:end]
        returns = self.replay_buffer.discounted_returns[path_ind][start:end]
        dones = self.replay_buffer.terminals[path_ind][start:end]
        ep_returns = self.replay_buffer.episode_returns[path_ind][start:end]
        batch = FlowTFBatch(
            trajectories, conditions, observations, actions, rewards, next_observations, returns, ep_returns, dones)
        return batch

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
    dataset = SequenceDataset(env_name="ant_dir")
