
import numpy as np


def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ToyReplayBuffer():
    def __init__(self, argus):
        self.argus = argus
        self._count = 0
        self.max_return, self.min_return = -9999999, 9999999
        self.max_ep_reward, self.min_ep_reward = -9999999, 9999999
        self._dict = {'path_lengths': [],}

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

    def _add_value_for_key(self):
        for key in self.keys:
            if key not in self._dict.keys():
                self._dict[key] = []

    @property
    def n_episodes(self):
        return self._count

    def add_path(self, path):
        ## if first path added, set keys based on contents
        self._add_keys(path)
        self._add_value_for_key()
        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            self._dict[key].append(array)

        path_length = len(path['energy'])
        self._dict['path_lengths'].append(path_length)
        ## increment path counter
        self._count += 1

    def finalize(self):
        self._add_attributes()
        # print(f'[ datasets_process/buffer ] Finalized replay buffer | {self._count} episodes')

    def items(self):
        return {k: v for k, v in self._dict.items() if k != 'path_lengths' and k != 'episode_returns'}.items()
        # return {k: v for k, v in self._dict.items() if k in ["observations", "actions"]}.items()