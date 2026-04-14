import pickle
import h5py
import collections
import numpy as np
import os
import xlrd
import xlwt
import pickle
from path_process.get_path import get_project_path

def consecutive_trajectory_2_separate_trajectory(dataset, max_episode_steps):
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    use_timeouts = 'timeouts' in dataset
    episode_step = 0
    for i in range(N):
        try:
            done_bool = bool(dataset['terminals'][i])
        except:
            done_bool = bool(dataset['dones'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == max_episode_steps - 1)

        for key in list(dataset.keys()):
            data_[key].append(np.expand_dims(dataset[key][i], axis=0))

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.vstack(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)
        episode_step += 1

def save_dataset(total_dataset_info, domain, save_path="dataset_analysis_result"):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(domain)
    write_to_which_row = 0
    keys_list = list(total_dataset_info[0].keys())
    for i, key in enumerate(keys_list):
        worksheet.write(write_to_which_row, i, key)
    write_to_which_row += 1
    for dataset_info in total_dataset_info:
        for i, (key, val) in enumerate(dataset_info.items()):
            worksheet.write(write_to_which_row, i, str(val))
        write_to_which_row += 1
    workbook.save(os.path.join(save_path, f"{domain}_offline_dataset_analysis.xls"))

def save_as_pickle(data, file_name, save_path="dataset_analysis_result"):
    file_path = os.path.join(save_path, f'{file_name}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def dataset_analysis(domain, env_name, dataset_type, file_type):
    assert domain in ["continual_world", "ant_dir"]
    return_info = {f"{env_name}_ep_idx": [], f"{env_name}_ep_return": []}
    dataset_info = {}
    if domain == "continual_world":
        dataset_path = f"{get_project_path()}/data/{domain}/{env_name}/{env_name}-{dataset_type}.{file_type}"
        if file_type == "hdf5":
            file = h5py.File(dataset_path, "r")
            trajectories_keys = ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]
            trajectories = {}
            for data_key in trajectories_keys:
                data = file[data_key][:]
                if data_key == "dones":
                    data_key = "terminals"
                trajectories[data_key] = data
            file.close()
        elif file_type == "pkl":
            file = open(dataset_path, "rb")
            trajectories = pickle.load(file)
            trajectories_info = {}
            for key, val in trajectories.items():
                trajectories_info[key] = np.shape(val)
            print(trajectories_info)
            dataset_info.update({"trajectories_info": trajectories_info})
        else:
            raise NotImplementedError
        itr = consecutive_trajectory_2_separate_trajectory(dataset=trajectories, max_episode_steps=200)
        success_rate = []
        success_return, failure_return = [], []
        for i, episode in enumerate(itr):
            if np.sum(episode["successes"]) > 0.5:
                success_rate.append(1)
                success_return.append(np.sum(episode["rewards"]))
            else:
                success_rate.append(0)
                failure_return.append(np.sum(episode["rewards"]))
            return_info[f"{env_name}_ep_idx"].append(i)
            return_info[f"{env_name}_ep_return"].append(sum(episode["rewards"]))
    elif domain == "ant_dir":
        dataset_path = f"{get_project_path()}/data/{domain}/{env_name}-{dataset_type}.{file_type}"
        success_rate = []
        success_return, failure_return = [], []
        if file_type == "hdf5":
            file = h5py.File(dataset_path, "r")
            trajectories_keys = ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]
            trajectories = {}
            for data_key in trajectories_keys:
                data = file[data_key][:]
                if data_key == "dones":
                    data_key = "terminals"
                trajectories[data_key] = data
            file.close()
        elif file_type == "pkl":
            file = open(dataset_path, "rb")
            trajectories = pickle.load(file)
            trajectories_info = {}
            for i, episode in enumerate(trajectories):
                success_rate.append(1)
                success_return.append(np.sum(episode["rewards"]))
                return_info[f"{env_name}_ep_idx"].append(i)
                return_info[f"{env_name}_ep_return"].append(sum(episode["rewards"]))
                for key, val in episode.items():
                    if key not in list(trajectories_info.keys()):
                        trajectories_info[key] = np.shape(val)
                    else:
                        new_data_shape = np.shape(val)
                        trajectories_info[key] = [trajectories_info[key][shape_i]+new_data_shape[shape_i] for shape_i in range(len(new_data_shape))]
            print(f"trajectories_info: {trajectories_info}")
            dataset_info.update({"trajectories_info": trajectories_info})
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    print(f"task:{env_name}, datatype:{dataset_type}, ep_length:{len(success_rate)}, mean_success: {np.mean(success_rate)}")
    if len(success_return) > 0:
        print(f"success return:[{np.min(success_return)},{np.mean(success_return)},{np.max(success_return)}]")
    if len(failure_return) > 0:
        print(f"failure return:[{np.min(failure_return)},{np.mean(failure_return)},{np.max(failure_return)}]")
    dataset_info.update({"task": env_name, "file_type": file_type, "dataset_quality": dataset_type, "ep_length": len(success_rate), "mean_success": np.mean(success_rate), "mean_return": np.mean(success_return+failure_return)})
    if domain == "continual_world":
        analysis_interval = 1000
    elif domain == "ant_dir":
        analysis_interval = 200
    else:
        raise NotImplementedError
    xx = [int(_ * analysis_interval) for _ in range((len(success_return) + len(failure_return)) // analysis_interval)]
    xx.append(-1)
    range_analysis = list(zip(xx[:-1], xx[1:]))
    # range_analysis = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, -1)]
    for range_i in range_analysis:
        print(f"ep[{range_i[0]}:{range_i[1]}], mean_success: {np.mean(success_rate[range_i[0]:range_i[1]])}")
        dataset_info.update({f"ep[{range_i[0]}:{range_i[1]}]_mean_success": np.mean(success_rate[range_i[0]:range_i[1]])})
    return dataset_info, return_info


def analyse_cw_datasets():
    env_name = ["hammer-v1", "push-wall-v1", "faucet-close-v1", "push-back-v1", "stick-pull-v1", "handle-press-side-v1", "push-v1", "shelf-place-v1", "window-close-v1", "peg-unplug-side-v1",
                "pick-place-v1", "door-open-v1", "drawer-open-v1", "drawer-close-v1", "button-press-topdown-v1", "peg-insert-side-v1", "window-open-v1", "door-close-v1", "reach-wall-v1",
                "pick-place-wall-v1", "button-press-v1", "button-press-topdown-wall-v1", "button-press-wall-v1", "disassemble-v1", "plate-slide-v1", "plate-slide-side-v1", "plate-slide-back-v1",
                "plate-slide-back-side-v1", "handle-press-v1", "handle-pull-v1", "handle-pull-side-v1", "stick-push-v1", "basketball-v1", "soccer-v1", "faucet-open-v1", "coffee-push-v1",
                "coffee-pull-v1", "coffee-button-v1", "sweep-v1", "sweep-into-v1", "pick-out-of-hole-v1", "assembly-v1", "lever-pull-v1", "dial-turn-v1"]
    dataset_type = ["expert"]
    file_type = ["pkl"]
    domain = "continual_world"

    total_return_info = []
    total_dataset_info = []
    for env_name_i in env_name:
        for dataset_type_i in dataset_type:
            for file_type_i in file_type:
                dataset_info, return_info = dataset_analysis(domain, env_name_i, dataset_type_i, file_type_i)
                total_dataset_info.append(dataset_info)
                total_return_info.append(return_info)
    print(total_return_info)
    save_dataset(total_dataset_info=total_dataset_info, domain=domain)
    save_as_pickle(data=total_return_info, file_name="total_return_info")
    save_as_pickle(data=total_dataset_info, file_name="total_dataset_info")

def analyse_ant_dir_datasets():
    env_name = ["4", "6", "7", "9", "10", "13", "15", "16", "17", "18", "19", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49"]
    dataset_type = ["expert"]
    file_type = ["pkl"]
    domain = "ant_dir"
    total_return_info = []
    total_dataset_info = []
    for env_name_i in env_name:
        env_name_i = f"ant_dir-{env_name_i}"
        for dataset_type_i in dataset_type:
            for file_type_i in file_type:
                dataset_info, return_info = dataset_analysis(domain, env_name_i, dataset_type_i, file_type_i)
                total_dataset_info.append(dataset_info)
                total_return_info.append(return_info)
    print(total_return_info)
    save_dataset(total_dataset_info=total_dataset_info, domain=domain)
    save_as_pickle(data=total_return_info, file_name="total_return_info")
    save_as_pickle(data=total_dataset_info, file_name="total_dataset_info")

if __name__ == "__main__":
    analyse_cw_datasets()
