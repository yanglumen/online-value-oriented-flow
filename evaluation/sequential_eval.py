import numpy as np
import time
from trainer.trainer_util import to_torch, to_np
from datasets_process.dataset_util import get_multi_task_name
import copy
from datasets_process.dataset_util import load_environment
import torch

def get_obs_with_task(domain, env_name, eval_env):
    if domain == 'ant_dir':
        if env_name == 'ant_dir':
            obs = eval_env.reset()
        else:
            raise NotImplementedError
    elif domain == 'gym':
        # if env_name in ['Pendulum-v0', "LunarLanderContinuous-v2"]:
        obs = eval_env.reset()
        # else:
        #     raise NotImplementedError
    else:
        raise Exception("Domain not supported")
    return obs

def check_observation_unnormalization(argus, dataset, observations):
    if argus.train_with_normed_data:
        observations = dataset.normalizer.unnormalize(observations, "observations")
    return observations

def check_observation_normalization(argus, dataset, observations):
    if argus.train_with_normed_data:
        observations = dataset.normalizer.normalize(observations, "observations")
    return observations

def sequential_evaluation(argus, dataset, model, current_ep, current_step, eval_episodes=30, random_action=False):
    ep_reward = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_env = dataset.eval_env
    start_time = time.time()
    for eval_i in range(eval_episodes):
        eval_env.seed(np.random.randint(0, 999))
        obs = get_obs_with_task(domain=argus.domain, env_name=dataset.env_name, eval_env=eval_env)
        while True:
            action = to_np(model.get_eval_action(to_torch(check_observation_normalization(argus=argus, dataset=dataset, observations=obs), device=argus.device)))
            action = np.clip(action, dataset.original_action_range[0], dataset.original_action_range[1])
            obs, reward, done, _ = eval_env.step(action)
            t[eval_i] = t[eval_i] + 1
            ep_reward[eval_i] = ep_reward[eval_i] + reward
            if done or t[eval_i] >= eval_env.max_episode_steps:
                break
        if eval_i % 10 == 0:
            print(f"Completing {eval_i} episode. Time Consumption: {time.time() - start_time}")
    eval_results = {f"{get_multi_task_name(argus)}_mean_ep_return": np.mean(ep_reward), f"{get_multi_task_name(argus)}_std_ep_return": np.std(ep_reward)}
    print_eval_results = " | ".join([f"{key}:{val}" for key, val in eval_results.items()])
    print(f"Ep: {current_ep} | Step: {current_step} | Time Consumption: {time.time() - start_time} | time_limit: {eval_env.max_episode_steps} |mean eval step: {np.mean(t)} | evaluation_info: {print_eval_results}")
    return eval_results


def parallel_ant_dir_eval(argus, dataset, model, specific_eta, specific_returns, eval_episodes=30, ddim_sample=True):
    start_time = time.time()
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    avg_score = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_envs = [load_environment(env_name=dataset.env_name, domain=argus.domain) for _ in range(eval_episodes)]
    for env_i in range(eval_episodes):
        eval_envs[env_i].seed(np.random.randint(0, 999))
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    env_info.update({"observations": obs})
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action = to_np(model.get_action(
                cond={0: to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device)},
                data_shape=(eval_episodes, argus.sequence_length, argus.input_channels), ddim_sample=ddim_sample,
                clip_range=dataset.data_range["normed_observations_range"], returns=specific_returns,
                specific_eta=specific_eta,
            ))
            # action = dataset.normalizer.unnormalize(x=action[:, 0, -argus.action_dim:], key="actions")
            action = action[:, 0, -argus.action_dim:]
        else:
            action = to_np(model.get_action(
                cond={0: to_torch(obs, device=argus.device)},
                data_shape=(eval_episodes, argus.sequence_length, argus.input_channels), ddim_sample=ddim_sample,
                clip_range=dataset.data_range["observations_range"], returns=specific_returns,
                specific_eta=specific_eta,
            ))
            action = action[:, 0, -argus.action_dim:]
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                obs_i, reward_i, done_i, _ = eval_envs[env_i].step(action[env_i])
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= dataset.max_path_length:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 20 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    try:
        for env_i in range(eval_episodes):
            avg_score[env_i] += eval_envs[env_i].get_normalized_score(avg_reward[env_i])
    except:
        pass
    eval_results.update({f"{argus.domain}_{argus.dataset}_ave_return": np.mean(avg_reward),
                         f"{argus.domain}_{argus.dataset}_ave_time_step": np.mean(t),
                         f"{argus.domain}_{argus.dataset}_ave_score": np.mean(avg_score)})
    print(f"Time Consumption: {time.time() - start_time}, {eval_results}")
    return eval_results

def parallel_d4rl_eval_score_function_version(
        argus, dataset, model, critic, guidance_scale, eval_episodes=30, ddim_sample=True):
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    avg_score = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_envs = dataset.eval_envs
    # eval_envs = [load_environment(env_name=dataset.env_name, domain=argus.domain) for _ in range(eval_episodes)]
    for env_i in range(eval_episodes):
        # eval_envs[env_i].seed(np.random.randint(0, 999))
        eval_envs[env_i].seed(1001 + env_i)
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    executed_actions = torch.randn((eval_episodes, argus.action_dim), device=argus.device).clamp(-argus.max_action_val, argus.max_action_val)
    env_info.update({"observations": obs})
    start_time = time.time()
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action = to_np(model.gen_action(
                states=to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device),
                critic=critic, steps=argus.flow_step, x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            ))
        else:
            action = to_np(model.gen_action(
                states=to_torch(obs, device=argus.device), critic=critic, steps=argus.flow_step,
                x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            ))
        executed_actions = to_torch(action, device=argus.device)
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                obs_i, reward_i, done_i, _ = eval_envs[env_i].step(np.clip(action[env_i], -0.999999, 0.999999))
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= eval_envs[env_i].max_episode_steps:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 50 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    try:
        for env_i in range(eval_episodes):
            avg_score[env_i] += eval_envs[env_i].get_normalized_score(avg_reward[env_i])
    except:
        pass
    eval_results.update({
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_return": np.mean(avg_reward),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_time_step": np.mean(t),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_score": np.mean(avg_score),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_std_score": np.std(avg_score)}
    )
    print(f"Time Consumption: {time.time() - start_time}, seed: {argus.seed}, {eval_results}")
    return eval_results

def parallel_d4rl_eval_adaptive_flow_step(
        argus, dataset, model, critic, guidance_scale, eval_episodes=30, ddim_sample=True):
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    avg_score = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_envs = dataset.eval_envs
    # eval_envs = [load_environment(env_name=dataset.env_name, domain=argus.domain) for _ in range(eval_episodes)]
    for env_i in range(eval_episodes):
        # eval_envs[env_i].seed(np.random.randint(0, 999))
        eval_envs[env_i].seed(1001 + env_i)
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    executed_actions = torch.randn((eval_episodes, argus.action_dim), device=argus.device).clamp(-argus.max_action_val, argus.max_action_val)
    env_info.update({"observations": obs})
    start_time = time.time()
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action = to_np(model.gen_action(
                states=to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device),
                critic=critic, steps=argus.eval_flow_step, x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            ))
        else:
            action = to_np(model.gen_action(
                states=to_torch(obs, device=argus.device), critic=critic, steps=argus.eval_flow_step,
                x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            ))
        executed_actions = to_torch(action, device=argus.device)
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                obs_i, reward_i, done_i, _ = eval_envs[env_i].step(np.clip(action[env_i], -0.999999, 0.999999))
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= eval_envs[env_i].max_episode_steps:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 50 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    try:
        for env_i in range(eval_episodes):
            avg_score[env_i] += eval_envs[env_i].get_normalized_score(avg_reward[env_i])
    except:
        pass
    eval_results.update({
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_return": np.mean(avg_reward),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_time_step": np.mean(t),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_score": np.mean(avg_score),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_std_score": np.std(avg_score)}
    )
    print(f"Time Consumption: {time.time() - start_time}, seed: {argus.seed}, {eval_results}")
    return eval_results

def parallel_d4rl_eval_behavior_flow(
        argus, dataset, model, critic, guidance_scale, eval_episodes=30, ddim_sample=True):
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    avg_score = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_envs = dataset.eval_envs
    # eval_envs = [load_environment(env_name=dataset.env_name, domain=argus.domain) for _ in range(eval_episodes)]
    for env_i in range(eval_episodes):
        # eval_envs[env_i].seed(np.random.randint(0, 999))
        eval_envs[env_i].seed(1001 + env_i)
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    executed_actions = torch.randn((eval_episodes, argus.action_dim), device=argus.device).clamp(-argus.max_action_val, argus.max_action_val)
    env_info.update({"observations": obs})
    start_time = time.time()
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action = to_np(model.behavior_action(
                states=to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device),
                critic=critic, steps=argus.flow_step, x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            ))
        else:
            action = to_np(model.behavior_action(
                states=to_torch(obs, device=argus.device), critic=critic, steps=argus.flow_step,
                x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            ))
        executed_actions = to_torch(action, device=argus.device)
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                obs_i, reward_i, done_i, _ = eval_envs[env_i].step(np.clip(action[env_i], -0.999999, 0.999999))
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= eval_envs[env_i].max_episode_steps:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 50 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    try:
        for env_i in range(eval_episodes):
            avg_score[env_i] += eval_envs[env_i].get_normalized_score(avg_reward[env_i])
    except:
        pass
    eval_results.update({
        f"behavior_flow_ave_return": np.mean(avg_reward),
        f"behavior_flow_ave_time_step": np.mean(t),
        f"behavior_flow_ave_score": np.mean(avg_score),
        f"behavior_flow_std_score": np.std(avg_score)}
    )
    print(f"Time Consumption: {time.time() - start_time}, seed: {argus.seed}, {eval_results}")
    return eval_results

def visualization_d4rl(argus, dataset, model, critic, guidance_scale, eval_episodes=30, ddim_sample=True):
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    avg_score = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    visualization_with_Q_value = []
    eval_envs = dataset.eval_envs
    # eval_envs = [load_environment(env_name=dataset.env_name, domain=argus.domain) for _ in range(eval_episodes)]
    for env_i in range(eval_episodes):
        # eval_envs[env_i].seed(np.random.randint(0, 999))
        eval_envs[env_i].seed(1001 + env_i)
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    executed_actions = torch.randn((eval_episodes, argus.action_dim), device=argus.device).clamp(-argus.max_action_val, argus.max_action_val)
    env_info.update({"observations": obs})
    start_time = time.time()
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action, intermediate_values = model.gen_action_and_Q_values(
                states=to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device),
                critic=critic, steps=argus.flow_step, x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            )
            action = to_np(action)
            intermediate_values = to_np(intermediate_values)
        else:
            action, intermediate_values = model.gen_action_and_Q_values(
                states=to_torch(obs, device=argus.device), critic=critic, steps=argus.flow_step,
                x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            )
            action = to_np(action)
            intermediate_values = to_np(intermediate_values)
        visualization_with_Q_value.append(intermediate_values)
        executed_actions = to_torch(action, device=argus.device)
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                obs_i, reward_i, done_i, _ = eval_envs[env_i].step(np.clip(action[env_i], -0.999999, 0.999999))
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= eval_envs[env_i].max_episode_steps:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 50 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    try:
        for env_i in range(eval_episodes):
            avg_score[env_i] += eval_envs[env_i].get_normalized_score(avg_reward[env_i])
    except:
        pass
    eval_results.update({
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_return": np.mean(avg_reward),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_time_step": np.mean(t),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_score": np.mean(avg_score),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_std_score": np.std(avg_score)}
    )
    print(f"Time Consumption: {time.time() - start_time}, seed: {argus.seed}, {eval_results}")
    return eval_results, visualization_with_Q_value


def parallel_d4rl_eval_transformer_version(
        argus, dataset, model, critic, guidance_scale, eval_episodes=30, ddim_sample=True):
    start_time = time.time()
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    avg_score = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_envs = dataset.eval_envs
    # eval_envs = [load_environment(env_name=dataset.env_name, domain=argus.domain) for _ in range(eval_episodes)]
    for env_i in range(eval_episodes):
        # eval_envs[env_i].seed(np.random.randint(0, 999))
        eval_envs[env_i].seed(1001 + env_i)
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    obs_sequence = []
    executed_actions = torch.randn((eval_episodes, argus.action_dim), device=argus.device).clamp(-argus.max_action_val, argus.max_action_val)
    env_info.update({"observations": obs})
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            obs_sequence.append(to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device))
            # action = to_np(model.gen_action(
            #     states=to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device),
            #     critic=critic, steps=argus.flow_step, x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            # ))
        else:
            obs_sequence.append(to_torch(obs, device=argus.device))
            # action = to_np(model.gen_action(
            #     states=to_torch(obs, device=argus.device), critic=critic, steps=argus.flow_step,
            #     x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            # ))
        action = to_np(model.gen_action(
            states=torch.stack(obs_sequence[-argus.sequence_length:], dim=1), rtg=argus.target_rtg,
            steps=argus.flow_step, x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
        ))
        executed_actions = to_torch(action, device=argus.device)
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                obs_i, reward_i, done_i, _ = eval_envs[env_i].step(np.clip(action[env_i], -0.999999, 0.999999))
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= eval_envs[env_i].max_episode_steps:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 50 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    try:
        for env_i in range(eval_episodes):
            avg_score[env_i] += eval_envs[env_i].get_normalized_score(avg_reward[env_i])
    except:
        pass
    eval_results.update({
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_return": np.mean(avg_reward),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_time_step": np.mean(t),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_score": np.mean(avg_score),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_std_score": np.std(avg_score)}
    )
    print(f"Time Consumption: {time.time() - start_time}, seed: {argus.seed}, {eval_results}")
    return eval_results

def flow2better_d4rl_eval(
        argus, dataset, model, inverse_dynamics, critic, guidance_scale, eval_episodes=30, ddim_sample=True):
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    avg_score = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_envs = dataset.eval_envs
    # eval_envs = [load_environment(env_name=dataset.env_name, domain=argus.domain) for _ in range(eval_episodes)]
    for env_i in range(eval_episodes):
        # eval_envs[env_i].seed(np.random.randint(0, 999))
        eval_envs[env_i].seed(1001 + env_i)
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    executed_actions = torch.randn((eval_episodes, argus.action_dim), device=argus.device).clamp(-argus.max_action_val, argus.max_action_val)
    env_info.update({"observations": obs})
    start_time = time.time()
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action = to_np(model.gen_action(
                states=to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device),
                inverse_dynamics=inverse_dynamics, critic=critic, steps=argus.flow_step,
                x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            ))
        else:
            action = to_np(model.gen_action(
                states=to_torch(obs, device=argus.device),
                inverse_dynamics=inverse_dynamics, critic=critic, steps=argus.flow_step,
                x_t_clip_value=argus.x_t_clip_value, executed_actions=executed_actions,
            ))
        executed_actions = to_torch(action, device=argus.device)
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                obs_i, reward_i, done_i, _ = eval_envs[env_i].step(np.clip(action[env_i], -0.999999, 0.999999))
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= eval_envs[env_i].max_episode_steps:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 50 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    try:
        for env_i in range(eval_episodes):
            avg_score[env_i] += eval_envs[env_i].get_normalized_score(avg_reward[env_i])
    except:
        pass
    eval_results.update({
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_return": np.mean(avg_reward),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_time_step": np.mean(t),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_ave_score": np.mean(avg_score),
        f"{argus.domain}_{argus.dataset}_GuiSca_{guidance_scale}_std_score": np.std(avg_score)}
    )
    print(f"Time Consumption: {time.time() - start_time}, seed: {argus.seed}, {eval_results}")
    return eval_results