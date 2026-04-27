import math

import numpy as np

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:
    def __init__(self, termination_penalty):
        self._dict = {
            'path_lengths': [],
        }
        self._count = 0
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets_process/buffer ] Info:\n' + '\n'.join(
            f'    {key}: [{np.sum(self.path_lengths)}, {np.shape(val[0])[-1]}]'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items() if k != 'path_lengths'}.items()

    def _add_value_for_key(self):
        for key in self.keys:
            if key not in self._dict.keys():
                self._dict[key] = []

    def add_path(self, path):
        path_length = len(path['observations'])
        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())
        ## if first path added, set keys based on contents
        self._add_keys(path)
        self._add_value_for_key()

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            self._dict[key].append(array)

        try:
            ## penalize early termination
            if path['terminals'].any() and self.termination_penalty is not None:
                assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
                self._dict['rewards'][self._count][-1] += self.termination_penalty
        except:
            pass

        ## record path length
        self._dict['path_lengths'].append(path_length)

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        self._add_attributes()
        print(f'[ datasets_process/buffer ] Finalized replay buffer | {self._count} episodes')

class ReturnReplayBuffer(ReplayBuffer):
    def __init__(self, argus, termination_penalty, discounts, max_path_length):
        super().__init__(termination_penalty=termination_penalty)
        self._dict = {
            'path_lengths': [],
            'discounted_returns': [],
            'episode_returns': [],
        }
        self.argus = argus
        self._count = 0
        self.termination_penalty = termination_penalty
        self.discounts = discounts ** np.arange(max_path_length)
        self.max_return, self.min_return = -9999999, 9999999
        self.max_ep_reward, self.min_ep_reward = -9999999, 9999999
        self.returns_separation = []

    def add_fake_path(self, path):
        for key in path.keys():
            if key not in list(self.keys):
                self.keys.append(key)
                self._dict[key] = []
            array = atleast_2d(path[key])
            self._dict[key].append(array)

    def add_fake_data(self, fake_data):
        for key in fake_data.keys():
            if key not in list(self.keys):
                self.keys.append(key)
                assert len(fake_data[key]) == self._count
                self._dict[key] = fake_data[key]
        self._add_attributes()

    def get_fake_paths(self):
        return {"fake_actions": self._dict["fake_actions"], "fake_next_actions": self._dict["fake_next_actions"]}

    def add_path(self, path):
        path_length = len(path['observations'])
        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())
        ## if first path added, set keys based on contents
        self._add_keys(path)
        self._add_value_for_key()
        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            self._dict[key].append(array)
        ## penalize early termination
        try:
            if path['terminals'].any() and self.termination_penalty is not None:
                assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
                self._dict['rewards'][self._count][-1] += self.termination_penalty
        except:
            pass
        ## record path length
        self._dict['path_lengths'].append(path_length)
        path_discounted_returns, path_undiscounted_returns = [], []
        for start_i in range(path_length):
            rewards = path["rewards"][start_i:]
            discounted_returns = (self.discounts[:len(rewards)] * np.squeeze(rewards)).sum()
            # discounted_returns = 1
            undiscounted_returns = rewards.sum()
            if discounted_returns > self.max_return: self.max_return = discounted_returns
            if discounted_returns < self.min_return: self.min_return = discounted_returns
            if undiscounted_returns > self.max_ep_reward: self.max_ep_reward = undiscounted_returns
            if undiscounted_returns < self.min_ep_reward: self.min_ep_reward = undiscounted_returns
            # discounted_returns = np.array([discounted_returns / self.returns_scale], dtype=np.float32)
            path_discounted_returns.append(discounted_returns)
            path_undiscounted_returns.append(undiscounted_returns)
        self._dict['discounted_returns'].append(np.reshape(np.array(path_discounted_returns), [-1, 1]))
        self._dict['episode_returns'].append(np.reshape(np.array(path_undiscounted_returns), [-1, 1]))
        self.max_return = math.floor(self.max_return*1000)/1000
        self.min_return = math.floor(self.min_return*1000)/1000
        ## increment path counter
        self._count += 1

    def check_returns_separation(self, x):
        returns_separation_value = 0
        for i in range(len(self.returns_separation) - 1):
            if self.returns_separation[i] < x:
                returns_separation_value = i
            else:
                break
        return self.argus.multi_etas[returns_separation_value], returns_separation_value
    def sophisticated_returns_separation(self):
        discounted_returns = np.vstack(self._dict['discounted_returns'])
        discounted_returns = np.sort(discounted_returns, axis=0)
        self.returns_separation.append(float(discounted_returns[0]))
        for i in range(len(self.argus.multi_etas)):
            self.returns_separation.append(float(discounted_returns[int((i+1)/len(self.argus.multi_etas) * len(discounted_returns))-1]))

    def return_normalization(self):
        for path_i in range(len(self._dict['discounted_returns'])):
            # self._dict['discounted_returns'][path_i] = (self._dict['discounted_returns'][path_i] - self.min_return) / (self.max_return - self.min_return)
            self._dict['discounted_returns'][path_i] = self._dict['discounted_returns'][path_i] / self.argus.returns_scale

    def reward_normalization(self):
        if self.argus.dataset == "walker2d-medium-replay-v2":
            for path_i in range(self._count):
                self._dict['rewards'][path_i] /= (self.max_ep_reward - self.min_ep_reward)
                self._dict['rewards'][path_i] *= 1000

    def finalize(self):
        # self.return_normalization()
        # self.reward_normalization()
        self.sophisticated_returns_separation()
        self._add_attributes()
        # print(f'[ datasets_process/buffer ] Finalized replay buffer | {self._count} episodes')

    def items(self):
        return {k: v for k, v in self._dict.items() if k != 'path_lengths' and k != 'episode_returns'}.items()
        # return {k: v for k, v in self._dict.items() if k in ["observations", "actions"]}.items()

class FlowTFReplayBuffer(ReturnReplayBuffer):
    def __init__(self, argus, termination_penalty, discounts, max_path_length):
        super().__init__(argus=argus, termination_penalty=termination_penalty, discounts=discounts, max_path_length=max_path_length)
        self._dict = {
            'path_lengths': [],
            'discounted_returns': [],
            'episode_returns': [],
        }
        self.argus = argus
        self._count = 0
        self.termination_penalty = termination_penalty
        self.discounts = discounts ** np.arange(max_path_length)
        self.max_return, self.min_return = -9999999, 9999999
        self.max_ep_return, self.min_ep_return = -9999999, 9999999
        self.returns_separation = []

    def add_path(self, path):
        path_length = len(path['observations'])
        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())
        ## if first path added, set keys based on contents
        self._add_keys(path)
        self._add_value_for_key()
        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            self._dict[key].append(array)
        ## penalize early termination
        try:
            if path['terminals'].any() and self.termination_penalty is not None:
                assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
                self._dict['rewards'][self._count][-1] += self.termination_penalty
        except:
            pass
        ## record path length
        self._dict['path_lengths'].append(path_length)
        path_discounted_returns, path_undiscounted_returns = [], []
        for start_i in range(path_length):
            rewards = path["rewards"][start_i:]
            discounted_returns = (self.discounts[:len(rewards)] * np.squeeze(rewards)).sum()
            # discounted_returns = 1
            undiscounted_returns = rewards.sum()
            if discounted_returns > self.max_return: self.max_return = discounted_returns
            if discounted_returns < self.min_return: self.min_return = discounted_returns
            if undiscounted_returns > self.max_ep_return: self.max_ep_return = undiscounted_returns
            if undiscounted_returns < self.min_ep_return: self.min_ep_return = undiscounted_returns
            # discounted_returns = np.array([discounted_returns / self.returns_scale], dtype=np.float32)
            path_discounted_returns.append(discounted_returns)
            path_undiscounted_returns.append(undiscounted_returns)
        self._dict['discounted_returns'].append(np.reshape(np.array(path_discounted_returns), [-1, 1]))
        self._dict['episode_returns'].append(np.reshape(np.array(path_undiscounted_returns), [-1, 1]))

        ## increment path counter
        self._count += 1

    def return_normalization(self):
        for path_i in range(len(self._dict['discounted_returns'])):
            self._dict['discounted_returns'][path_i] = (self._dict['discounted_returns'][path_i] - self.min_return) / (self.max_return - self.min_return)
            self._dict['episode_returns'][path_i] = (self._dict['episode_returns'][path_i] - self.min_ep_return) / (self.max_ep_return - self.min_ep_return)
            # self._dict['discounted_returns'][path_i] = self._dict['discounted_returns'][path_i] / self.argus.returns_scale


class OnlineReturnReplayBuffer(ReplayBuffer):
    def __init__(self, argus, termination_penalty, discounts, max_path_length, buffer_keys, normalize_keys):
        super().__init__(termination_penalty=termination_penalty)
        self._dict = {
            'path_lengths': [],
            'discounted_returns': [],
            'episode_returns': [],
        }
        for key in buffer_keys:
            self._dict.update({key: []})
        self.argus = argus
        self._count = 0
        self.termination_penalty = termination_penalty
        self.discounts = discounts ** np.arange(max_path_length)
        self.max_return, self.min_return = -9999999, 9999999
        self.normalize_keys = normalize_keys
        self._add_attributes()

    def _check_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            if not hasattr(self, key):
                setattr(self, key, val)

    def add_path(self, path):
        path_length = len(path['observations'])
        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())
        ## if first path added, set keys based on contents
        self._add_keys(path)
        self._add_value_for_key()
        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            self._dict[key].append(array)
        ## penalize early termination
        try:
            if path['terminals'].any() and self.termination_penalty is not None:
                assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
                self._dict['rewards'][self._count][-1] += self.termination_penalty
        except:
            pass
        ## record path length
        self._dict['path_lengths'].append(path_length)
        path_discounted_returns, path_undiscounted_returns = [], []
        for start_i in range(path_length):
            rewards = path["rewards"][start_i:]
            discounted_returns = (self.discounts[:len(rewards)] * np.squeeze(rewards)).sum()
            undiscounted_returns = rewards.sum()
            if discounted_returns > self.max_return: self.max_return = discounted_returns
            if discounted_returns < self.min_return: self.min_return = discounted_returns
            # discounted_returns = np.array([discounted_returns / self.returns_scale], dtype=np.float32)
            path_discounted_returns.append(discounted_returns)
            path_undiscounted_returns.append(undiscounted_returns)
        self._dict['discounted_returns'].append(np.reshape(np.array(path_discounted_returns), [-1, 1]))
        self._dict['episode_returns'].append(np.reshape(np.array(path_undiscounted_returns), [-1, 1]))

        ## increment path counter
        self._count += 1
        if self._count == 1:
            self._check_attributes()

    def _compute_path_returns(self, path):
        path_length = len(path['observations'])
        path_discounted_returns, path_undiscounted_returns = [], []
        for start_i in range(path_length):
            rewards = path["rewards"][start_i:]
            discounted_returns = (self.discounts[:len(rewards)] * np.squeeze(rewards)).sum()
            undiscounted_returns = rewards.sum()
            path_discounted_returns.append(discounted_returns)
            path_undiscounted_returns.append(undiscounted_returns)
        return (
            np.reshape(np.array(path_discounted_returns), [-1, 1]),
            np.reshape(np.array(path_undiscounted_returns), [-1, 1]),
        )

    def _refresh_return_extrema(self):
        if len(self._dict['discounted_returns']) == 0:
            self.max_return, self.min_return = -9999999, 9999999
            return
        discounted_returns = np.vstack(self._dict['discounted_returns'])
        self.max_return = float(np.max(discounted_returns))
        self.min_return = float(np.min(discounted_returns))

    def replace_path(self, path_ind, path):
        path_length = len(path['observations'])
        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())
        self._add_keys(path)
        self._add_value_for_key()

        for key in self.keys:
            array = atleast_2d(path[key])
            self._dict[key][path_ind] = array

        try:
            if path['terminals'].any() and self.termination_penalty is not None:
                assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
                self._dict['rewards'][path_ind][-1] += self.termination_penalty
        except:
            pass

        self._dict['path_lengths'][path_ind] = path_length
        discounted_returns, undiscounted_returns = self._compute_path_returns(path)
        self._dict['discounted_returns'][path_ind] = discounted_returns
        self._dict['episode_returns'][path_ind] = undiscounted_returns
        self._refresh_return_extrema()

    def __repr__(self):
        return '[ datasets_process/buffer ] Info:\n' + '\n'.join(
            f'    {key}: [{np.sum(self.path_lengths)}, (None, None)] because of initial buffer setting'
            for key, val in self.items()
        )

# class TrajectoryBuffer():
#     def __init__(self, buffer_keys):
#         self._dict = {}
#         self._count = 0
#         for key in buffer_keys:
#             self._dict.update({key: []})
#
#     def

