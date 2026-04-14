import pickle
# from continualworld.envs import get_mt50
import h5py
import numpy as np
import collections

def get_d4rl_dataset(env):
    dataset = env.get_dataset()
    return dataset

def get_mujoco_dataset_iter(envs, task_idx):
    dataset = get_d4rl_dataset(envs[task_idx])
    N = dataset['rewards'].shape[0]
    min_reward = np.min(dataset['rewards'])
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == envs[task_idx]._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            if k == 'infos/action_log_probs':
                data_['action_log_probs'].append(dataset[k][i])
            else:
                data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data, min_reward
            data_ = collections.defaultdict(list)

        episode_step += 1

def load_offline_datasets(project_path, env_name, task_idx, continual_world_dataset_quality="expert", continual_world_data_type="hdf5", envs=None):
    if env_name=="cheetah_vel":
        raise NotImplementedError
    elif env_name=="ant_dir":
        assert task_idx in ["4", "6", "7", "9", "10", "13", "15", "16", "17", "18", "19", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                            "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49"]
        dataset_path = f"{project_path}/data/{env_name}/{env_name}-{task_idx}-expert.pkl"
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        return trajectories
    elif env_name == "cheetah_dir":
        assert task_idx in ["0", "1"]
        dataset_path = f"{project_path}/data/{env_name}/{env_name}-{task_idx}-expert.pkl"
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        return trajectories
    elif env_name == "ML1-pick-place-v2":
        assert task_idx in ["3", "4", "5", "6", "7", "8", "28", "29", "35", "37", "38", "42", "43", "45", "46", "47", "48", "49"]
        dataset_path = f"{project_path}/data/{env_name}/{env_name}-{task_idx}-expert.pkl"
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        return trajectories
    elif env_name == "continual_world":
        assert task_idx in [
            "hammer-v1", "push-wall-v1", "faucet-close-v1", "push-back-v1", "stick-pull-v1", "handle-press-side-v1", "push-v1", "shelf-place-v1", "window-close-v1", "peg-unplug-side-v1",
            "pick-place-v1", "door-open-v1", "drawer-open-v1", "drawer-close-v1", "button-press-topdown-v1", "peg-insert-side-v1", "window-open-v1", "door-close-v1", "reach-wall-v1",
            "pick-place-wall-v1", "button-press-v1", "button-press-topdown-wall-v1", "button-press-wall-v1", "disassemble-v1", "plate-slide-v1", "plate-slide-side-v1", "plate-slide-back-v1",
            "plate-slide-back-side-v1", "handle-press-v1", "handle-pull-v1", "handle-pull-side-v1", "stick-push-v1", "basketball-v1", "soccer-v1", "faucet-open-v1", "coffee-push-v1",
            "coffee-pull-v1", "coffee-button-v1", "sweep-v1", "sweep-into-v1", "pick-out-of-hole-v1", "assembly-v1", "push-back-v1", "lever-pull-v1", "dial-turn-v1",
            # =========================================================================================================
            "hammer-v2", "push-wall-v2", "faucet-close-v2", "push-back-v2", "stick-pull-v2", "handle-press-side-v2",
            "push-v2", "shelf-place-v2", "window-close-v2", "peg-unplug-side-v2",
        ]
        dataset_path = f"{project_path}/data/{env_name}/{task_idx}/{task_idx}-{continual_world_dataset_quality}.{continual_world_data_type}"
        if continual_world_data_type == "hdf5":
            file = h5py.File(dataset_path, "r")
            trajectories_keys = ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]
            trajectories = {}
            for data_key in trajectories_keys:
                data = file[data_key][:]
                if data_key == "dones":
                    data_key = "terminals"
                trajectories[data_key] = data
            file.close()
        elif continual_world_data_type == "pkl":
            file = open(dataset_path, "rb")
            trajectories = pickle.load(file)
            trajectories["terminals"] = trajectories["dones"]
            file.close()
        else:
            raise NotImplementedError
        return trajectories
    elif env_name == "d4rl":
        assert envs is not None
        trajectories = []
        data_iter = get_mujoco_dataset_iter(envs=envs, task_idx=task_idx)
        for i, episode in enumerate(data_iter):
            episode_data, min_reward = episode
            trajectories.append(episode_data)
        return trajectories
    else:
        raise NotImplementedError



