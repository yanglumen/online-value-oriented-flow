import collections
import pickle

import numpy as np

try:
    import gymnasium as modern_gym
except ImportError:
    modern_gym = None

try:
    import gym as legacy_gym
except ImportError:
    legacy_gym = None

from config.multistep_rl_flow_hyperparameter import RLTrainMode
from path_process.get_path import get_project_path


def _require_modern_gym():
    if modern_gym is None:
        raise ImportError(
            "gymnasium is required for the online Mujoco pipeline. "
            "Install the project with uv sync to get gymnasium[mujoco]."
        )
    return modern_gym


def _require_legacy_gym():
    if legacy_gym is None:
        raise ImportError(
            "gym is required for the legacy D4RL pipeline. "
            "Install the optional offline dependencies if you still need D4RL."
        )
    return legacy_gym


def _require_d4rl():
    try:
        import d4rl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "D4RL is not installed. The offline dataset pipeline is optional in the new "
            "Python 3.10 setup. Install the optional offline extras only if you need D4RL."
        ) from exc


def _make_modern_env(env_name):
    gym_mod = _require_modern_gym()
    return gym_mod.make(env_name)


def _make_legacy_env(env_name):
    gym_mod = _require_legacy_gym()
    return gym_mod.make(env_name)


def _infer_max_episode_steps(env):
    for attr in ("_max_episode_steps", "max_episode_steps"):
        value = getattr(env, attr, None)
        if value is not None:
            return value
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None) is not None:
        return spec.max_episode_steps
    return None


def _finalize_env(wrapped_env, env_name, original_env=False):
    env = wrapped_env if original_env else wrapped_env.unwrapped
    max_episode_steps = _infer_max_episode_steps(wrapped_env)
    if max_episode_steps is not None:
        env.max_episode_steps = max_episode_steps
        env._max_episode_steps = max_episode_steps
    env.name = env_name
    return env


def load_environment(env_name, domain=None, original_env=False, eval_env=False):
    del eval_env
    if not isinstance(env_name, str):
        return env_name

    domain = "gymnasium" if domain in [None, "gym", "gymnasium"] else domain
    domain = "d4rl" if domain == "mujoco" else domain
    if domain == "gymnasium":
        wrapped_env = _make_modern_env(env_name)
        return _finalize_env(wrapped_env, env_name, original_env=original_env)
    if domain == "d4rl":
        _require_d4rl()
        wrapped_env = _make_legacy_env(env_name)
        return _finalize_env(wrapped_env, env_name, original_env=original_env)
    raise NotImplementedError(f"Unsupported domain: {domain}")


def get_environment_info(env_name, domain, original_env=False, task_idx="0"):
    if domain in ["gym", "gymnasium"]:
        env = _finalize_env(_make_modern_env(env_name), env_name, original_env=original_env)
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_range = [float(np.min(env.action_space.low)), float(np.max(env.action_space.high))]
    elif domain == "ant_dir":
        from datasets_process.multi_mujoco_env.mujoco_control_envs import AntDirEnv

        assert isinstance(task_idx, str) and 0 <= int(task_idx) <= 49
        tasks = []
        project_path = get_project_path()
        with open(
            f"{project_path}/datasets_process/multi_mujoco_env/ant_dir_config/config_ant_dir_task{task_idx}.pkl",
            "rb",
        ) as f:
            task_info = pickle.load(f)
            tasks.append(task_info[0])
        env = AntDirEnv(tasks, len(tasks), include_goal=False)
        env.max_episode_steps = env._max_episode_steps
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_range = [float(env.action_space.low), float(env.action_space.high)]
    elif domain in ["d4rl", "mujoco"]:
        env = load_environment(env_name, domain=domain, original_env=original_env)
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_range = [float(np.min(env.action_space.low)), float(np.max(env.action_space.high))]
    else:
        raise NotImplementedError(f"Unsupported domain: {domain}")
    return env, observation_dim, action_dim, action_range


def get_multi_task_name(argus):
    postfix, prefix = [], []
    multi_task_name = []
    multi_task_name.append(f"{argus.domain}_{argus.dataset}")
    if len(prefix) > 0:
        multi_task_name.insert(0, prefix[0])
    if len(postfix) > 0:
        multi_task_name.append(postfix[0])
    multi_task_name = "-".join(multi_task_name)
    return multi_task_name


def maze2d_set_terminals_legacy(env, dataset):
    env = load_environment(env, domain="d4rl") if isinstance(env, str) else env
    goal = np.array(env._target)
    threshold = 0.5

    xy = dataset["observations"][:, :2]
    distances = np.linalg.norm(xy - goal, axis=-1)
    at_goal = distances < threshold
    timeouts = np.zeros_like(dataset["timeouts"])
    timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

    timeout_steps = np.where(timeouts)[0]
    path_lengths = timeout_steps[1:] - timeout_steps[:-1]

    print(
        f"[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | "
        f"min length: {path_lengths.min()} | max length: {path_lengths.max()}"
    )

    dataset["timeouts"] = timeouts
    return dataset


def maze2d_set_terminals(env, dataset):
    env = load_environment(env, domain="d4rl") if isinstance(env, str) else env
    goal = np.array(env._target)
    threshold = 0.5

    data_length = len(dataset["rewards"])
    xy = dataset["observations"][:, :2]
    distances = np.linalg.norm(xy - goal, axis=-1)
    at_goal = distances < threshold
    timeouts = np.zeros_like(dataset["timeouts"])
    timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]
    timeout_steps = np.where(timeouts)[0]
    env_name_body = "-".join(env.name.split("-")[1:-1])
    if env_name_body in ["large-dense", "umaze-dense"]:
        for i in timeout_steps:
            dataset["terminals"][i] = True
        timeout_steps = np.concatenate([timeout_steps, np.array([data_length])], axis=-1)
        timeout_idx = 0
        idx = 0
        while idx < data_length:
            if (timeout_steps[timeout_idx] - idx) > env.max_episode_steps:
                idx += env.max_episode_steps
            else:
                idx = timeout_steps[timeout_idx]
                timeout_idx += 1
                if timeout_idx >= len(timeout_steps):
                    timeout_idx = -1
            if idx >= data_length:
                break
            timeouts[idx] = True
        timeout_steps = np.where(timeouts)[0]
    elif env_name_body not in ["large", "medium", "umaze", "medium-dense"]:
        raise NotImplementedError

    path_lengths = timeout_steps[1:] - timeout_steps[:-1]

    print(
        f"[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | "
        f"min length: {path_lengths.min()} | max length: {path_lengths.max()}"
    )

    dataset["timeouts"] = timeouts
    return dataset


def antmaze_set_terminals(env, dataset):
    env = load_environment(env, domain="d4rl") if isinstance(env, str) else env
    timeout_steps = np.where(dataset["timeouts"])[0]
    path_lengths = timeout_steps[1:] - timeout_steps[:-1]
    print(
        f"[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | "
        f"min length: {path_lengths.min()} | max length: {path_lengths.max()}"
    )
    return dataset


def antmaze_episode_length_statistics(env, dataset):
    timeout_steps = np.where(dataset["timeouts"])[0]
    path_lengths = timeout_steps[1:] - timeout_steps[:-1]
    print(
        f"[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | "
        f"min length: {path_lengths.min()} | max length: {path_lengths.max()}"
    )
    return dataset


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def get_d4rl_dataset(env, rl_mode, reward_tune="iql_antmaze"):
    _require_d4rl()
    import d4rl

    if "antmaze" in str(env).lower():
        dataset = d4rl.qlearning_dataset(env)
        reward = np.zeros_like(dataset["rewards"])
        if reward_tune == "normalize":
            dataset["rewards"] = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            dataset["rewards"] = reward - 1.0
        elif reward_tune == "iql_locomotion":
            min_ret, max_ret = return_range(dataset, 1000)
            reward /= (max_ret - min_ret)
            dataset["rewards"] *= 1000
        elif reward_tune == "cql_antmaze":
            dataset["rewards"] = (reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            dataset["rewards"] = (reward - 0.25) * 2.0
        elif reward_tune == "AEPO":
            dataset["rewards"] = reward - 0.1
        elif reward_tune == "no_tune":
            dataset["rewards"] = dataset["rewards"] * 10
    elif env.name.split("-")[0] in ["hammer", "pen", "relocate", "door"]:
        dataset = d4rl.qlearning_dataset(env)
        dataset["rewards"] = (dataset["rewards"] - dataset["rewards"].min()) / (
            dataset["rewards"].max() - dataset["rewards"].min()
        )
    else:
        dataset = env.get_dataset()
    if "maze2d" in str(env).lower():
        if rl_mode == RLTrainMode.grpo:
            dataset = maze2d_set_terminals_legacy(env, dataset)
        else:
            dataset = maze2d_set_terminals(env, dataset)
        if "dense" not in str(env).lower():
            dataset["rewards"] = dataset["rewards"] - 0.1
    return dataset


def d4rl_trajectories_iterator(env, reward_tune="iql_antmaze", rl_mode=RLTrainMode.grpo, CEP_dataset_load_mode=False):
    _require_d4rl()
    import d4rl

    if CEP_dataset_load_mode:
        dataset = d4rl.qlearning_dataset(env)
        reward = dataset["rewards"]
        if reward_tune == "normalize":
            dataset["rewards"] = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            dataset["rewards"] = reward - 1.0
        elif reward_tune == "iql_locomotion":
            min_ret, max_ret = return_range(dataset, 1000)
            reward /= (max_ret - min_ret)
            dataset["rewards"] *= 1000
        elif reward_tune == "cql_antmaze":
            dataset["rewards"] = (reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            dataset["rewards"] = (reward - 0.25) * 2.0
        elif reward_tune == "AEPO":
            dataset["rewards"] = reward - 0.1
    else:
        dataset = get_d4rl_dataset(env, rl_mode, reward_tune)

    data_ = collections.defaultdict(list)
    use_timeouts = "timeouts" in dataset
    episode_step = 0
    min_reward = np.min(dataset["rewards"])
    for i in range(dataset["rewards"].shape[0]):
        episode_step += 1
        done_bool = bool(dataset["terminals"][i])
        final_timestep = dataset["timeouts"][i] if use_timeouts else (episode_step == env.max_episode_steps)

        for k in dataset:
            if "metadata" in k:
                continue
            if k == "infos/action_log_probs":
                data_["action_log_probs"].append(dataset[k][i])
            else:
                data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            if "antmaze" not in env.name and "maze2d" not in env.name and episode_step > env.max_episode_steps:
                raise Exception("Episode steps reached")
            episode_step = 0
            episode_data = {k: np.array(v) for k, v in data_.items()}
            if "maze2d" in env.name:
                env_name_body = "-".join(env.name.split("-")[1:-1])
                if env_name_body in ["large-dense", "umaze-dense"]:
                    episode_data = process_maze2d_episode(episode_data)
                elif env_name_body in ["large", "medium", "umaze", "medium-dense"]:
                    episode_data = process_maze2d_episode_legacy(episode_data)
                else:
                    raise NotImplementedError
            yield episode_data, min_reward
            data_ = collections.defaultdict(list)


def process_maze2d_episode(episode):
    assert "next_observations" not in episode
    next_observations = np.concatenate(
        [episode["observations"][1:].copy(), episode["observations"][-1:].copy()], axis=0
    )
    episode["next_observations"] = next_observations
    return episode


def process_maze2d_episode_legacy(episode):
    assert "next_observations" not in episode
    next_observations = episode["observations"][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode["next_observations"] = next_observations
    return episode


def process_antmaze_episode(episode):
    assert "next_observations" not in episode
    next_observations = np.concatenate(
        [episode["observations"][1:].copy(), episode["observations"][-1:].copy()], axis=0
    )
    episode["next_observations"] = next_observations
    return episode
