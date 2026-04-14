import torch
import numpy as np
from path_process.get_path import get_project_path
from datasets_process.normalizer import DatasetNormalizer
from collections import namedtuple
from termcolor import colored
import time
from toy_example.toy_dataset import inf_train_gen
from toy_example.dataset_process.toy_buffer import ToyReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns dones')
ToyBatch = namedtuple('Batch', 'datas energy')

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, argus=None, project_path=None, env_name='ant_dir'):
        if project_path is None:
            project_path = get_project_path()
        self.sequence_length = 1
        self.project_path = project_path
        self.argus = argus
        self.env_name = env_name
        self.replay_buffer = self.get_data_from_dataset()
        self.indices = self.make_indices(self.replay_buffer.path_lengths)
        self.observation_dim = self.replay_buffer.datas[0].shape[-1]
        self.action_dim = self.replay_buffer.datas[0].shape[-1]
        self.transition_dim = self.observation_dim + self.action_dim
        self.n_episodes = self.replay_buffer.n_episodes
        self.path_lengths = self.replay_buffer.path_lengths
        self.max_path_length = np.max(self.replay_buffer.path_lengths)
        print(self.replay_buffer)

    def get_data_from_dataset(self):
        datas, energy = inf_train_gen(self.argus.dataset, batch_size=self.argus.datanum)
        replay_buffer = ToyReplayBuffer(argus=self.argus)
        replay_buffer.add_path(path={"datas": datas, "energy": energy})
        replay_buffer.finalize()
        return replay_buffer

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
        datas = self.replay_buffer.datas[path_ind][start:end]
        energy = self.replay_buffer.energy[path_ind][start:end]
        batch = ToyBatch(datas, energy)
        return batch