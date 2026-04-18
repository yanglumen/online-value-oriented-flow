import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import wandb

from config.dict2class import obj2dict
from config.multistep_rl_flow_hyperparameter import CriticType, RLTrainMode, WeightedSamplesType
from trainer.shared_flow_rl_core import (
    adv_policy_update,
    adv_value_update,
    behavior_flow_update,
    energy_critic_update,
    generate_behavior_flow_action,
    generate_train_flow_action,
)
from trainer.trainer_util import batch_to_device, to_torch, to_np


class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class online_guided_flow_trainer(object):
    def __init__(
            self, argus, train_flow, target_train_flow,
            behavior_flow, flow_energy_model, energy_model, dataset):
        self.argus = argus
        self.model = train_flow
        self.model.to(self.argus.device)
        self.target_model = target_train_flow
        self.target_model.to(self.argus.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.behavior_flow = behavior_flow
        self.behavior_flow.to(self.argus.device)
        self.flow_energy_model = flow_energy_model
        self.flow_energy_model.to(self.argus.device)
        self.energy_model = energy_model
        self.energy_model.to(self.argus.device)
        self.dataset = dataset
        self.device = argus.device
        self.loss_fn = torch.nn.MSELoss()
        self.ema = EMA(argus.ema_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=argus.lr)
        self.bf_optimizer = torch.optim.Adam(self.behavior_flow.parameters(), lr=argus.lr)
        if self.argus.rl_mode in [RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl3, 
                                  RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
            self.q_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
            self.fv_v_optimizer = torch.optim.Adam(self.flow_energy_model.flow_v.parameters(), lr=argus.lr)
        elif self.argus.rl_mode in [RLTrainMode.flow_constrained_rl, RLTrainMode.direct_flow_2_result]:
            self.q_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
        else:
            self.flow_value_func = self.flow_energy_model.v
            self.fv_optimizer = torch.optim.Adam(self.flow_energy_model.q.parameters(), lr=argus.lr)
            self.fv_v_optimizer = torch.optim.Adam(self.flow_energy_model.v.parameters(), lr=argus.lr)
        self.energy_q_optimizer = torch.optim.Adam(self.energy_model.q.parameters(), lr=argus.lr)
        if argus.critic_type in [CriticType.iql, CriticType.ciql, CriticType.cql]:
            self.energy_v_optimizer = torch.optim.Adam(self.energy_model.v.parameters(), lr=argus.lr)
        elif argus.critic_type in [CriticType.isql]:
            self.energy_v_optimizer = None
        else:
            raise NotImplementedError

        self.step = 0
        self.flow_update_count = 0
        self.train_flow_initialized_from_behavior = False
        self.env_step = 0
        self.save_path = argus.save_path
        self.save_freq = argus.save_freq
        self.env_name = argus.dataset
        self.wandb_log = argus.wandb_log
        self.wandb_exp_name = argus.wandb_exp_name
        self.wandb_exp_group = argus.wandb_exp_group
        self.wandb_log_frequency = argus.wandb_log_frequency
        self.online_print_frequency = getattr(argus, "online_print_frequency", 1000)
        self.wandb_project_name = argus.wandb_project_name
        self.wandb_mode = getattr(argus, "wandb_mode", None)
        self.wandb_init_timeout = getattr(argus, "wandb_init_timeout", 120)
        self.best_model_info = {"guidance_scale": 0.0, "performance": -1e9, "performance_std": 0.0}
        if self.argus.rl_mode in [RLTrainMode.adv_rl, RLTrainMode.grpo] and self.argus.sequence_length < 2:
            raise ValueError(
                f"rl_mode={self.argus.rl_mode} requires sequence_length >= 2 for transition-based flow updates. "
                f"Got sequence_length={self.argus.sequence_length}."
            )

        self.collect_env = dataset.env
        self.collect_obs = self._reset_env(self.collect_env)
        self.prev_action = torch.zeros((1, self.argus.action_dim), device=self.argus.device)
        self.current_path = self._empty_path()
        self.current_episode_steps = 0

        if self.wandb_log:
            wandb_kwargs = {
                "name": self.wandb_exp_name,
                "group": self.wandb_exp_group,
                "project": self.wandb_project_name,
                "config": obj2dict(self.argus),
                "settings": wandb.Settings(init_timeout=self.wandb_init_timeout),
            }
            if self.wandb_mode:
                wandb_kwargs["mode"] = self.wandb_mode
            wandb.init(**wandb_kwargs)
            wandb.define_metric("env_step")
            wandb.define_metric("replay_episodes")
            wandb.define_metric("replay_steps")
            wandb.define_metric("online_rollout_return")
            wandb.define_metric("online_rollout_length")
            wandb.define_metric("rollout_policy_type")
            wandb.define_metric("rollout_policy_random_frac")
            wandb.define_metric("rollout_policy_behavior_frac")
            wandb.define_metric("rollout_policy_train_flow_frac")
            wandb.define_metric("online_eval_return_det")
            wandb.define_metric("online_eval_return_stoch")
            wandb.define_metric("critic_train_minus_behavior_q_mean")
            wandb.define_metric("flow_behavior_action_l2_mean")
            wandb.define_metric(f"{self.argus.domain}_{self.argus.dataset}_online_eval_return")
            wandb.define_metric(f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_std")
            wandb.define_metric(f"{self.argus.domain}_{self.argus.dataset}_online_eval_length")
            wandb.define_metric(f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_det")
            wandb.define_metric(f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_stoch")

    def _compact_train_metrics(self, train_stats, loss):
        metrics = {}
        for key in [
            "env_step",
            "replay_episodes",
            "replay_steps",
            "online_rollout_return",
            "online_rollout_length",
            "flow_update_count",
            "train_flow_initialized_from_behavior",
            "rollout_policy_id",
            "rollout_policy_type",
            "rollout_policy_random_frac",
            "rollout_policy_behavior_frac",
            "rollout_policy_train_flow_frac",
            "deploy_flow_allowed",
            "deploy_train_flow_prob",
            "online_action_noise_std_current",
            "policy_update_count",
            "critic_update_interval",
        ]:
            if key in train_stats:
                metrics[key] = train_stats[key]

        for key in [
            "q0_loss",
            "v0_loss",
            "behavior_flow_loss",
            "flow_value_loss",
            "flow_v_value_loss",
            "flow_divergence",
            "flow_loss",
            "flow_adv_mean",
            "flow_adv_std",
            "flow_behavior_action_l2_mean",
            "flow_behavior_action_l2_std",
            "train_action_clip_frac",
            "behavior_action_clip_frac",
            "critic_train_action_q_mean",
            "critic_behavior_action_q_mean",
            "critic_train_minus_behavior_q_mean",
            "critic_updated",
        ]:
            if key in loss:
                metrics[key] = loss[key]
        return metrics

    def _compact_eval_metrics(self, eval_results):
        keys = [
            "online_eval_return_det",
            "online_eval_return_stoch",
            "online_eval_length_det",
            "online_eval_length_stoch",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_std",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_length",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_det",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_det_std",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_length_det",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_stoch",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_stoch_std",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_length_stoch",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_policy_id",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_deploy_flow_allowed",
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_train_flow_prob",
        ]
        return {key: eval_results[key] for key in keys if key in eval_results}

    def _metric_str(self, metrics, key, precision=3):
        value = metrics.get(key)
        if value is None:
            return f"{key}=na"
        if isinstance(value, (float, int, np.floating, np.integer)):
            return f"{key}={float(value):.{precision}f}"
        return f"{key}={value}"

    def _format_train_summary(self, epoch, train_stats, loss, elapsed):
        parts = [
            f"Epoch {epoch}",
            f"EnvStep {self.env_step}",
            f"Step {self.step}",
            f"ReplayEpisodes {self.dataset.replay_buffer.n_episodes}",
            f"ReplaySteps {train_stats.get('replay_steps', 0)}",
            f"Time {elapsed:.2f}s",
        ]
        if "online_rollout_return" in train_stats:
            parts.append(f"RolloutReturn {train_stats['online_rollout_return']:.2f}")
        if "online_rollout_length" in train_stats:
            parts.append(f"RolloutLen {train_stats['online_rollout_length']:.1f}")
        if "flow_update_count" in train_stats:
            parts.append(f"FlowUpdates {int(train_stats['flow_update_count'])}")
        if "critic_update_interval" in train_stats:
            parts.append(f"CriticEvery {int(train_stats['critic_update_interval'])}")
        if "train_flow_initialized_from_behavior" in train_stats:
            parts.append(f"InitFromBF {int(train_stats['train_flow_initialized_from_behavior'])}")
        if "rollout_policy_id" in train_stats:
            parts.append(f"Policy {train_stats.get('rollout_policy_type', self._policy_id_to_type(train_stats['rollout_policy_id']))}")
        if "deploy_train_flow_prob" in train_stats:
            parts.append(f"TrainProb {train_stats['deploy_train_flow_prob']:.3f}")
        if loss:
            parts.extend([
                self._metric_str(loss, "q0_loss"),
                self._metric_str(loss, "v0_loss"),
                self._metric_str(loss, "behavior_flow_loss"),
                self._metric_str(loss, "flow_value_loss"),
                self._metric_str(loss, "flow_v_value_loss"),
                self._metric_str(loss, "flow_divergence"),
                self._metric_str(loss, "flow_loss"),
                self._metric_str(loss, "flow_behavior_action_l2_mean"),
                self._metric_str(loss, "critic_train_minus_behavior_q_mean"),
            ])
        else:
            parts.append("loss=empty")
        return " | ".join(parts)

    def _format_eval_summary(self, eval_results):
        return " | ".join([
            "Online evaluation",
            f"EnvStep {self.env_step}",
            f"Step {self.step}",
            f"DetReturn {eval_results['online_eval_return_det']:.2f}",
            f"StochReturn {eval_results['online_eval_return_stoch']:.2f}",
            f"DetLen {eval_results['online_eval_length_det']:.1f}",
            f"StochLen {eval_results['online_eval_length_stoch']:.1f}",
            f"Best {self.best_model_info['performance']:.2f}",
        ])

    def _normalize_advantage(self, advantage):
        # TODO(online/offline): This remains for online GRPO only. The shared
        # adv_rl policy update intentionally uses the offline raw-advantage loss.
        if not getattr(self.argus, "grpo_adv_batch_norm", True):
            return advantage
        adv_mean = advantage.mean().detach()
        adv_std = advantage.std(unbiased=False).detach()
        return (advantage - adv_mean) / (adv_std + 1e-6)

    def _empty_path(self):
        return {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "next_observations": [],
            "dones": [],
            "terminals": [],
        }

    def _reset_env(self, env):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            return reset_result[0]
        return reset_result

    def _step_env(self, env, action):
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            return next_obs, reward, bool(terminated), bool(truncated), info
        next_obs, reward, done, info = step_result
        return next_obs, reward, bool(done), False, info

    @contextmanager
    def _policy_eval_mode(self):
        modules = [
            self.model,
            self.target_model,
            self.behavior_flow,
            self.flow_energy_model,
            self.energy_model,
        ]
        previous_modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, was_training in zip(modules, previous_modes):
                module.train(was_training)

    def _maybe_update_normalizer(self):
        if self.argus.train_with_normed_data and self.dataset.replay_buffer.n_episodes > 0:
            self.dataset.assign_normalizer_parameters()

    def _behavior_bootstrap_updates(self):
        configured_updates = int(getattr(self.argus, "online_behavior_bootstrap_updates", -1))
        if configured_updates >= 0:
            return configured_updates
        return max(1, int(self.argus.online_updates_per_epoch))

    def _should_use_random_policy(self):
        return self.env_step < self.argus.online_random_steps or self.step < self._behavior_bootstrap_updates()

    def _flow_time_gate_open(self):
        return self.step >= self.argus.update_flow_start_epoch * max(1, self.argus.online_updates_per_epoch)

    def _deploy_flow_allowed(self):
        if self.argus.online_behavior_only:
            return False
        if self._should_use_random_policy():
            return False
        deploy_after_updates = int(getattr(self.argus, "online_deploy_flow_after_updates", 0))
        deploy_after_epoch = int(getattr(self.argus, "online_deploy_flow_after_epoch", 0))
        deploy_after_step = deploy_after_epoch * max(1, self.argus.online_updates_per_epoch)
        min_flow_updates = int(getattr(self.argus, "online_min_flow_updates_before_deploy", 1))
        return (
            self.step >= deploy_after_step
            and self.flow_update_count >= deploy_after_updates
            and self.flow_update_count >= min_flow_updates
        )

    def _train_flow_deploy_prob(self):
        if not self._deploy_flow_allowed():
            return 0.0
        if not getattr(self.argus, "online_gradual_deploy_enable", False):
            return 1.0
        start_prob = float(getattr(self.argus, "online_gradual_deploy_start_prob", 0.1))
        end_prob = float(getattr(self.argus, "online_gradual_deploy_end_prob", 1.0))
        ramp_updates = max(1, int(getattr(self.argus, "online_gradual_deploy_ramp_updates", 5000)))
        deploy_after_updates = int(getattr(self.argus, "online_deploy_flow_after_updates", 0))
        progress = np.clip((self.flow_update_count - deploy_after_updates) / ramp_updates, 0.0, 1.0)
        return float(start_prob + (end_prob - start_prob) * progress)

    def _should_use_behavior_policy(self, deterministic=False):
        if self.argus.online_behavior_only:
            return True
        if not self._deploy_flow_allowed():
            return True
        train_flow_prob = self._train_flow_deploy_prob()
        if deterministic:
            return train_flow_prob < 0.5
        return np.random.rand() >= train_flow_prob

    def _current_rollout_policy_id(self):
        if self._should_use_random_policy():
            return 0
        if self._should_use_behavior_policy(deterministic=True):
            return 1
        if getattr(self.argus, "online_gradual_deploy_enable", False) and self._train_flow_deploy_prob() < 1.0:
            return 3
        return 2

    def _policy_id_to_type(self, policy_id):
        return {
            0: "random",
            1: "behavior",
            2: "train_flow",
            3: "mixed",
        }.get(int(policy_id), "unknown")

    def _rollout_policy_stats(self, policy_counts, rollout_steps):
        total_steps = max(1, int(rollout_steps))
        if policy_counts.get(1, 0) > 0 and policy_counts.get(2, 0) > 0:
            dominant_policy = 3
        else:
            dominant_policy = max(policy_counts, key=policy_counts.get)
        return {
            "rollout_policy_id": float(dominant_policy),
            "rollout_policy_type": self._policy_id_to_type(dominant_policy),
            "rollout_policy_random_frac": float(policy_counts.get(0, 0) / total_steps),
            "rollout_policy_behavior_frac": float(policy_counts.get(1, 0) / total_steps),
            "rollout_policy_train_flow_frac": float(policy_counts.get(2, 0) / total_steps),
        }

    def _current_action_noise_std(self):
        if not getattr(self.argus, "online_action_noise_decay_enable", False):
            return float(getattr(self.argus, "online_action_noise_std", 0.0))
        start_std = float(getattr(self.argus, "online_action_noise_start_std", 0.5))
        end_std = float(getattr(self.argus, "online_action_noise_end_std", 0.1))
        decay_steps = max(1, int(getattr(self.argus, "online_action_noise_decay_steps", 1000000)))
        progress = min(1.0, max(0.0, float(self.env_step) / decay_steps))
        if start_std <= 0.0 or end_std <= 0.0:
            return end_std + (start_std - end_std) * (1.0 - progress)
        return start_std * ((end_std / start_std) ** progress)

    def _apply_training_action_noise(self, action_tensor):
        # TODO(online/offline): Exploration noise is online-only. The replayed
        # action target may include this post-policy noise, unlike offline data.
        if not getattr(self.argus, "online_action_noise_enable", False):
            return action_tensor
        noise_std = self._current_action_noise_std()
        if noise_std <= 0:
            return action_tensor
        noise = torch.randn_like(action_tensor) * noise_std
        noise_clip = getattr(self.argus, "online_action_noise_clip", None)
        if noise_clip is not None:
            noise = noise.clamp(-float(noise_clip), float(noise_clip))
        return (action_tensor + noise).clamp(-self.argus.max_action_val, self.argus.max_action_val)

    def _maybe_init_train_flow_from_behavior(self):
        if not getattr(self.argus, "online_init_train_flow_from_behavior", True):
            return
        if self.train_flow_initialized_from_behavior:
            return
        self.model.load_state_dict(self.behavior_flow.state_dict())
        self.target_model.load_state_dict(self.model.state_dict())
        self.train_flow_initialized_from_behavior = True

    def _sample_policy_tensor(self, obs_tensor, prev_action, deterministic=False, return_policy_id=False):
        with self._policy_eval_mode():
            with torch.no_grad():
                if self._should_use_behavior_policy(deterministic=deterministic):
                    policy_id = 1
                    action_tensor = generate_behavior_flow_action(
                        behavior_flow=self.behavior_flow, states=obs_tensor,
                        config=self.argus, deterministic=deterministic)
                else:
                    policy_id = 2
                    action_tensor = generate_train_flow_action(
                        flow_model=self.model, critic=self.flow_energy_model,
                        states=obs_tensor, previous_actions=prev_action,
                        config=self.argus, deterministic=deterministic)
        if return_policy_id:
            return action_tensor, policy_id
        return action_tensor

    def _policy_diagnostics(self, observations):
        if observations.shape[0] == 0:
            return {}
        diag_batch = observations[:min(256, observations.shape[0])]
        prev_action = torch.zeros((diag_batch.shape[0], self.argus.action_dim), device=self.argus.device)
        with self._policy_eval_mode():
            with torch.no_grad():
                behavior_action = generate_behavior_flow_action(
                    behavior_flow=self.behavior_flow, states=diag_batch,
                    config=self.argus, deterministic=True)
                train_action = generate_train_flow_action(
                    flow_model=self.model, critic=self.flow_energy_model,
                    states=diag_batch, previous_actions=prev_action,
                    config=self.argus, deterministic=True)
                action_diff = torch.norm(train_action - behavior_action, dim=-1)
                clip_margin = 1e-4
                train_clip_frac = (
                    train_action.abs() >= (self.argus.max_action_val - clip_margin)
                ).float().mean()
                behavior_clip_frac = (
                    behavior_action.abs() >= (self.argus.max_action_val - clip_margin)
                ).float().mean()
                train_q = self.energy_model.get_scaled_q(obs=diag_batch, act=train_action, scale=1.0)
                behavior_q = self.energy_model.get_scaled_q(obs=diag_batch, act=behavior_action, scale=1.0)
        return {
            "flow_behavior_action_l2_mean": action_diff.mean().detach().cpu().numpy().item(),
            "flow_behavior_action_l2_std": action_diff.std(unbiased=False).detach().cpu().numpy().item(),
            "train_action_clip_frac": train_clip_frac.detach().cpu().numpy().item(),
            "behavior_action_clip_frac": behavior_clip_frac.detach().cpu().numpy().item(),
            "critic_train_action_q_mean": train_q.mean().detach().cpu().numpy().item(),
            "critic_behavior_action_q_mean": behavior_q.mean().detach().cpu().numpy().item(),
            "critic_train_minus_behavior_q_mean": (train_q - behavior_q).mean().detach().cpu().numpy().item(),
        }

    def _policy_action(self, obs):
        if self._should_use_random_policy():
            action = self.collect_env.action_space.sample()
            self.prev_action = to_torch(np.asarray(action, dtype=np.float32)[None], device=self.argus.device)
            return np.asarray(action, dtype=np.float32), 0.0, 0

        obs_batch = np.asarray(obs, dtype=np.float32)[None]
        if self.argus.train_with_normed_data:
            obs_batch = self.dataset.normalizer.normalize(obs_batch, "observations")
        obs_tensor = to_torch(obs_batch, device=self.argus.device)
        action_tensor, policy_id = self._sample_policy_tensor(
            obs_tensor, self.prev_action, deterministic=False, return_policy_id=True)
        action_tensor = self._apply_training_action_noise(action_tensor)
        self.prev_action = action_tensor.detach()
        return to_np(action_tensor.squeeze(0)).astype(np.float32), 0.0, policy_id

    def collect_rollouts(self, rollout_steps):
        episode_returns = []
        episode_lengths = []
        policy_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(rollout_steps):
            action, log_prob, policy_id = self._policy_action(self.collect_obs)
            policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
            next_obs, reward, terminated, truncated, _ = self._step_env(self.collect_env, action)
            self.current_episode_steps += 1
            timed_out = truncated or self.current_episode_steps >= self.collect_env.max_episode_steps
            episode_done = terminated or timed_out
            self.current_path["observations"].append(np.asarray(self.collect_obs, dtype=np.float32))
            self.current_path["actions"].append(np.asarray(action, dtype=np.float32))
            self.current_path["log_probs"].append(np.asarray([log_prob], dtype=np.float32))
            self.current_path["rewards"].append(np.asarray([reward], dtype=np.float32))
            self.current_path["next_observations"].append(np.asarray(next_obs, dtype=np.float32))
            self.current_path["dones"].append(np.asarray([float(terminated)], dtype=np.float32))
            self.current_path["terminals"].append(np.asarray([bool(terminated)], dtype=bool))
            self.collect_obs = next_obs
            self.env_step += 1

            if episode_done:
                path = {key: np.asarray(val) for key, val in self.current_path.items()}
                ep_return = float(np.sum(path["rewards"]))
                ep_length = int(len(path["rewards"]))
                self.dataset.store_trajectories([path])
                self._maybe_update_normalizer()
                episode_returns.append(ep_return)
                episode_lengths.append(ep_length)
                self.collect_obs = self._reset_env(self.collect_env)
                self.prev_action = torch.zeros((1, self.argus.action_dim), device=self.argus.device)
                self.current_path = self._empty_path()
                self.current_episode_steps = 0
        return episode_returns, episode_lengths, self._rollout_policy_stats(policy_counts, rollout_steps)

    def sample_batch(self):
        if self.dataset.__len__(indices_type="ac") < self.argus.batch_size:
            return None
        batch = self.dataset.sample_trajectories(batch_size=self.argus.batch_size, indices_type="ac")
        return batch_to_device(batch, device=self.device, convert_to_torch_float=True)

    def _train_step(self, epoch):
        batch = self.sample_batch()
        if batch is None:
            return {}
        observation_seq = batch.observations
        action_seq = batch.actions
        next_observation_seq = batch.next_observations
        reward_seq = batch.rewards
        done_seq = batch.dones

        observations = observation_seq.reshape(-1, self.argus.observation_dim)
        actions = action_seq.reshape(-1, self.argus.action_dim)
        next_observations = next_observation_seq.reshape(-1, self.argus.observation_dim)
        rewards = reward_seq.reshape(-1, 1)
        dones = done_seq.reshape(-1, 1)

        valid_next_mask = (1.0 - done_seq[:, :-1].reshape(-1, 1)).bool().squeeze(-1)
        # TODO(online/offline): Online still builds transition batches with a done
        # mask from sequence replay, while offline adv training uses adjacent rows.
        transition_observations = observation_seq[:, :-1].reshape(-1, self.argus.observation_dim)[valid_next_mask]
        transition_actions = action_seq[:, :-1].reshape(-1, self.argus.action_dim)[valid_next_mask]
        transition_next_observations = next_observation_seq[:, :-1].reshape(-1, self.argus.observation_dim)[valid_next_mask]
        transition_next_actions = action_seq[:, 1:].reshape(-1, self.argus.action_dim)[valid_next_mask]
        if self.argus.debug_mode:
            update_energy = True
            update_behavior = True
            update_flow = True
        else:
            update_energy = True
            update_behavior = True
            update_flow = epoch >= self.argus.update_flow_start_epoch
        if self.argus.online_behavior_only:
            # TODO(online/offline): Behavior-only is an online deployment baseline,
            # not a shared offline learning-mode semantic.
            update_energy = False
            update_flow = False
            update_behavior = True
        critic_update_interval = max(1, int(getattr(self.argus, "online_critic_update_interval", 1)))
        adv_policy_step_available = (
            update_flow
            and self.argus.rl_mode == RLTrainMode.adv_rl
            and transition_observations.shape[0] > 0
        )
        update_adv_critic = (
            not update_flow
            or self.argus.rl_mode != RLTrainMode.adv_rl
            or (
                adv_policy_step_available
                and self.flow_update_count % critic_update_interval == 0
            )
        )
        if update_energy and self.argus.rl_mode == RLTrainMode.adv_rl:
            update_energy = update_adv_critic
        loss = {}
        if update_energy:
            loss.update(self.energy_train(
                iql_tau=self.argus.iql_tau, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones))
        if update_behavior:
            loss.update(self.behavior_flow_train(observations=observations, actions=actions))
        if update_flow:
            self._maybe_init_train_flow_from_behavior()
            # TODO(online/offline): Flow deployment/init gating remains online-only;
            # only the learning step below is shared.
            if self.argus.rl_mode == RLTrainMode.use_rl_q:
                loss.update(self.flow_train(observations=observations, actions=actions))
            elif self.argus.rl_mode == RLTrainMode.adv_rl:
                if transition_observations.shape[0] > 0:
                    if update_adv_critic:
                        loss.update(self.adv_based_value_train(
                            observations=transition_observations, actions=transition_actions))
                    loss.update(self.adv_based_policy_train(
                        observations=transition_observations, actions=transition_actions))
                    loss["critic_updated"] = float(update_adv_critic)
                    self.flow_update_count += 1
            elif self.argus.rl_mode == RLTrainMode.grpo:
                if transition_observations.shape[0] > 0:
                    loss.update(self.grpo_value_train(
                        observations=transition_observations, actions=transition_actions))
                    loss.update(self.grpo_policy_train(
                        observations=transition_observations, actions=transition_actions,
                        next_observations=transition_next_observations, next_actions=transition_next_actions,
                        expectile=self.argus.gfpo_expectile))
                    self.flow_update_count += 1
            elif self.argus.rl_mode == RLTrainMode.direct_flow_2_result:
                loss.update(self.direct_flow2result_value_train(
                    observations=observations, actions=actions, next_observations=next_observations, rewards=rewards, dones=dones))
                loss.update(self.direct_flow2result_policy_train(observations=observations, actions=actions))
            elif self.argus.rl_mode == RLTrainMode.full_rl_like:
                loss.update(self.full_rl_like_value_train(
                    observations=observations, actions=actions, next_observations=next_observations, rewards=rewards, dones=dones))
                loss.update(self.full_rl_like_policy_train(observations=observations, actions=actions))
            elif self.argus.rl_mode in [RLTrainMode.flow_constrained_rl, RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl3, RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
                loss.update(self.flow_constrained_value_train(
                    observations=observations, actions=actions, next_observations=next_observations, rewards=rewards, dones=dones))
                if transition_observations.shape[0] > 0:
                    loss.update(self.flow_constrained_policy_train(
                        observations=transition_observations, actions=transition_actions,
                        next_observations=transition_next_observations, next_actions=transition_next_actions))
            else:
                raise NotImplementedError(f"Unsupported online rl_mode: {self.argus.rl_mode}")
        loss.update(self._policy_diagnostics(observations=observations))
        return loss

    def online_train(self, num_epochs, rollout_steps_per_epoch, num_updates_per_epoch):
        try:
            if int(num_updates_per_epoch) != int(rollout_steps_per_epoch):
                print(
                    "Using one online update per environment step; "
                    f"overriding online_updates_per_epoch={num_updates_per_epoch} "
                    f"with rollout_steps_per_epoch={rollout_steps_per_epoch}."
                )
                num_updates_per_epoch = rollout_steps_per_epoch
                self.argus.online_updates_per_epoch = rollout_steps_per_epoch
            warmup_steps = max(self.argus.online_init_steps, self.argus.batch_size)
            if self.dataset.__len__(indices_type="ac") < warmup_steps:
                self.collect_rollouts(warmup_steps)
            last_log_time = time.time()
            for epoch in range(num_epochs):
                for _ in range(rollout_steps_per_epoch):
                    rollout_returns, rollout_lengths, rollout_policy_stats = self.collect_rollouts(1)
                    if rollout_returns:
                        train_stats = {
                            "online_rollout_return": float(np.mean(rollout_returns)),
                            "online_rollout_length": float(np.mean(rollout_lengths)),
                            "env_step": self.env_step,
                            "replay_episodes": self.dataset.replay_buffer.n_episodes,
                            "replay_steps": int(np.sum(self.dataset.replay_buffer.path_lengths)),
                            "flow_update_count": self.flow_update_count,
                            "policy_update_count": self.flow_update_count,
                            "critic_update_interval": max(1, int(getattr(self.argus, "online_critic_update_interval", 1))),
                            "train_flow_initialized_from_behavior": float(self.train_flow_initialized_from_behavior),
                            "deploy_flow_allowed": float(self._deploy_flow_allowed()),
                            "deploy_train_flow_prob": self._train_flow_deploy_prob(),
                            "online_action_noise_std_current": self._current_action_noise_std(),
                        }
                    else:
                        train_stats = {
                            "env_step": self.env_step,
                            "replay_episodes": self.dataset.replay_buffer.n_episodes,
                            "replay_steps": int(np.sum(self.dataset.replay_buffer.path_lengths)),
                            "flow_update_count": self.flow_update_count,
                            "policy_update_count": self.flow_update_count,
                            "critic_update_interval": max(1, int(getattr(self.argus, "online_critic_update_interval", 1))),
                            "train_flow_initialized_from_behavior": float(self.train_flow_initialized_from_behavior),
                            "deploy_flow_allowed": float(self._deploy_flow_allowed()),
                            "deploy_train_flow_prob": self._train_flow_deploy_prob(),
                            "online_action_noise_std_current": self._current_action_noise_std(),
                        }
                    train_stats.update(rollout_policy_stats)

                    loss = self._train_step(epoch)
                    if self.wandb_log and self.step % self.wandb_log_frequency == 0:
                        wandb.log(self._compact_train_metrics(train_stats, loss), step=self.step)
                    if self.step > 0 and self.step % self.argus.online_eval_freq == 0:
                        self.eval_online()
                    if self.step > 0 and self.step % self.save_freq == 0:
                        self.save_critic_checpoint()
                    if self.step % self.online_print_frequency == 0:
                        print(self._format_train_summary(epoch, train_stats, loss, time.time() - last_log_time))
                        last_log_time = time.time()
                    self.step += 1
        finally:
            if self.wandb_log and wandb.run is not None:
                wandb.finish()

    def _eval_online_once(self, deterministic):
        eval_returns = []
        eval_lengths = []
        for eval_idx in range(self.argus.eval_episodes):
            eval_env = self.dataset.eval_env
            obs = self._reset_env(eval_env)
            prev_action = torch.zeros((1, self.argus.action_dim), device=self.argus.device)
            ep_return = 0.0
            ep_length = 0
            while True:
                obs_batch = np.asarray(obs, dtype=np.float32)[None]
                if self.argus.train_with_normed_data and self.dataset.replay_buffer.n_episodes > 0:
                    obs_batch = self.dataset.normalizer.normalize(obs_batch, "observations")
                obs_tensor = to_torch(obs_batch, device=self.argus.device)
                action_tensor = self._sample_policy_tensor(
                    obs_tensor,
                    prev_action,
                    deterministic=deterministic,
                )
                action = to_np(action_tensor.squeeze(0))
                prev_action = action_tensor.detach()
                obs, reward, terminated, truncated, _ = self._step_env(eval_env, action)
                ep_return += float(reward)
                ep_length += 1
                if terminated or truncated or ep_length >= eval_env.max_episode_steps:
                    break
            eval_returns.append(ep_return)
            eval_lengths.append(ep_length)
        return {
            "return": float(np.mean(eval_returns)),
            "return_std": float(np.std(eval_returns)),
            "length": float(np.mean(eval_lengths)),
        }

    def eval_online(self):
        run_det = getattr(self.argus, "online_eval_deterministic", True)
        run_stoch = getattr(self.argus, "online_eval_stochastic", True)
        det_results = self._eval_online_once(deterministic=True) if run_det else None
        stoch_results = self._eval_online_once(deterministic=False) if run_stoch else None
        primary_results = det_results if det_results is not None else stoch_results
        eval_results = {
            "online_eval_return_det": det_results["return"] if det_results is not None else np.nan,
            "online_eval_return_stoch": stoch_results["return"] if stoch_results is not None else np.nan,
            "online_eval_length_det": det_results["length"] if det_results is not None else np.nan,
            "online_eval_length_stoch": stoch_results["length"] if stoch_results is not None else np.nan,
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return": primary_results["return"],
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_std": primary_results["return_std"],
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_length": primary_results["length"],
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_det": det_results["return"] if det_results is not None else np.nan,
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_det_std": det_results["return_std"] if det_results is not None else np.nan,
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_length_det": det_results["length"] if det_results is not None else np.nan,
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_stoch": stoch_results["return"] if stoch_results is not None else np.nan,
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_stoch_std": stoch_results["return_std"] if stoch_results is not None else np.nan,
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_length_stoch": stoch_results["length"] if stoch_results is not None else np.nan,
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_policy_id": float(self._current_rollout_policy_id()),
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_deploy_flow_allowed": float(self._deploy_flow_allowed()),
            f"{self.argus.domain}_{self.argus.dataset}_online_eval_train_flow_prob": self._train_flow_deploy_prob(),
        }
        if eval_results[f"{self.argus.domain}_{self.argus.dataset}_online_eval_return"] > self.best_model_info["performance"]:
            self.best_model_info["performance"] = eval_results[f"{self.argus.domain}_{self.argus.dataset}_online_eval_return"]
            self.best_model_info["performance_std"] = eval_results[f"{self.argus.domain}_{self.argus.dataset}_online_eval_return_std"]
            data = {
                "guidance_scale": self.best_model_info["guidance_scale"],
                "performance": self.best_model_info["performance"],
                "step": self.step,
                "flow": self.model.state_dict(),
            }
            savepath = os.path.join(self.get_detailed_save_path(), "best_flow_checkpoint")
            os.makedirs(savepath, exist_ok=True)
            torch.save(data, os.path.join(savepath, f"flow_{self.step}.pt"))
        print(self._format_eval_summary(eval_results))
        if self.wandb_log:
            wandb.log(self._compact_eval_metrics(eval_results), step=self.step)

    def get_detailed_save_path(self):
        return os.path.join(
            self.save_path, self.argus.mode, self.argus.domain, "_".join(self.env_name.split("-")), self.argus.current_exp_label)

    def energy_train(self, iql_tau, observations, actions, next_observations, rewards, dones, next_actions=None, fake_actions=None, fake_next_actions=None):
        return energy_critic_update(
            batch={
                "iql_tau": iql_tau,
                "observations": observations,
                "actions": actions,
                "next_observations": next_observations,
                "next_actions": next_actions,
                "rewards": rewards,
                "dones": dones,
                "fake_actions": fake_actions,
                "fake_next_actions": fake_next_actions,
            },
            models={
                "energy_model": self.energy_model,
                "behavior_flow": self.behavior_flow,
            },
            optimizers={
                "energy_q_optimizer": self.energy_q_optimizer,
                "energy_v_optimizer": self.energy_v_optimizer,
            },
            config=self.argus,
        )

    def behavior_flow_train(self, observations, actions):
        return behavior_flow_update(
            batch={"observations": observations, "actions": actions},
            models={
                "energy_model": self.energy_model,
                "behavior_flow": self.behavior_flow,
            },
            optimizers={"bf_optimizer": self.bf_optimizer},
            config=self.argus,
        )

    def flow_train(self, observations, actions):
        loss = {}
        loss.update(self.rl_like_flow_value_train(observations=observations, actions=actions))
        loss.update(self.rl_like_flow_policy_train(observations=observations, actions=actions))
        return loss

    def rl_like_flow_value_train(self, observations, actions):
        from models.rl_flow_forward_process import sample_weighted_interpolated_points

        x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        noise_x_t = (x_t + torch.randn_like(x_t)).clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        noise_dxdt = (dx_dt + torch.randn_like(dx_dt)).clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t)
        conservative_q_loss = torch.logsumexp(
            self.flow_energy_model.q(x=torch.cat([observations, noise_x_t, noise_dxdt], dim=-1), t=t), dim=0) - Q.mean()
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
        with torch.no_grad():
            target_fv = self.energy_model.get_scaled_q(obs=observations, act=actions, scale=self.argus.energy_scale)
        Q_loss = self.loss_fn(Q, target_fv.detach()) + self.argus.conservative_coef * conservative_q_loss.mean()
        V_loss = self.loss_fn(V, target_fv.detach())
        self.fv_optimizer.zero_grad()
        Q_loss.backward()
        self.fv_optimizer.step()
        self.fv_v_optimizer.zero_grad()
        V_loss.backward()
        self.fv_v_optimizer.step()
        self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        return {
            "flow_value_loss": Q_loss.detach().cpu().numpy().item(),
            "flow_v_value_loss": V_loss.detach().cpu().numpy().item(),
            "flow_value": Q.mean().detach().cpu().numpy().item(),
            "flow_v_value": V.mean().detach().cpu().numpy().item(),
        }

    def rl_like_flow_policy_train(self, observations, actions):
        from models.rl_flow_forward_process import sample_weighted_interpolated_points

        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        pred_u = pred_u.clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        cos_sim = torch.nn.functional.cosine_similarity(pred_u, behavior_u.detach(), dim=-1)
        divergence = 1 - cos_sim.unsqueeze(dim=-1)
        normed_pred_u = pred_u / torch.norm(pred_u, dim=-1, keepdim=True)
        pred_Q = self.flow_energy_model.q_target(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        pred_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_pred_u], dim=-1), t=t)
        loss = -(pred_Q - pred_V)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def flow_constrained_value_train(self, observations, actions, next_observations, rewards, dones):
        if self.argus.rl_mode == RLTrainMode.flow_constrained_rl2:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl2_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl3:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl3_loss(
                flow=self.model, behavior_flow=self.behavior_flow, energy_model=self.energy_model,
                observations=observations, actions=actions, next_observations=next_observations, rewards=rewards,
                dones=dones, tau=self.argus.iql_tau, multiple_actions=self.argus.flow_constrained_rl4_multiple_actions
            )
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl4:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl4_loss(
                flow=self.model, behavior_flow=self.behavior_flow, energy_model=self.energy_model,
                observations=observations, actions=actions, next_observations=next_observations, rewards=rewards,
                dones=dones, tau=self.argus.iql_tau, multiple_actions=self.argus.flow_constrained_rl4_multiple_actions,
            )
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl5:
            Q_loss, V_loss, flow_V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl5_loss(
                flow=self.model, behavior_flow=self.behavior_flow, energy_model=self.energy_model,
                observations=observations, actions=actions, next_observations=next_observations, rewards=rewards,
                dones=dones, tau=self.argus.iql_tau, multiple_actions=self.argus.flow_constrained_rl5_multiple_actions,
            )
            self.fv_v_optimizer.zero_grad()
            flow_V_loss.backward()
            self.fv_v_optimizer.step()
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl:
            Q_loss, V_loss, flow_loss_info = self.flow_energy_model.flow_constrained_rl_loss(
                flow=self.model, behavior_flow=self.behavior_flow, observations=observations, actions=actions,
                next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
        else:
            raise NotImplementedError
        if Q_loss is not None:
            self.q_optimizer.zero_grad()
            Q_loss.backward()
            self.q_optimizer.step()
            self.ema.update_model_average(self.flow_energy_model.q_target, self.flow_energy_model.q)
        if V_loss is not None:
            self.v_optimizer.zero_grad()
            V_loss.backward()
            self.v_optimizer.step()
        return flow_loss_info

    def flow_constrained_policy_train(self, observations, actions, next_observations, next_actions):
        from models.rl_flow_forward_process import sample_weighted_interpolated_points

        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=next_observations, actions=next_actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )

        pred_u = self.model(x=torch.cat([next_observations, x_t], dim=-1), t=t)
        dt = 1 / self.argus.flow_step
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([next_observations, x_t], dim=-1), t=t)
        divergence = torch.norm(pred_u - behavior_u.detach(), dim=-1, keepdim=True)
        x_t_plus_deltat = x_t + pred_u * dt
        x_t_plus_deltat = torch.clip(x_t_plus_deltat, -self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        if self.argus.rl_mode == RLTrainMode.flow_constrained_rl2:
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl4:
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl5:
            adv = (-self.argus.divergence_coef * divergence + self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, actions, next_observations], dim=-1))) - self.flow_energy_model.flow_v(torch.cat([x_t, actions, next_observations], dim=-1))
            loss = -adv
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl3:
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.flow_v(torch.cat([x_t_plus_deltat, next_observations], dim=-1))
        elif self.argus.rl_mode == RLTrainMode.flow_constrained_rl:
            loss = self.argus.divergence_coef * divergence - self.flow_energy_model.v(torch.cat([x_t_plus_deltat, next_observations], dim=-1))
        else:
            raise NotImplementedError
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def direct_flow2result_value_train(self, observations, actions, next_observations, rewards, dones):
        Q_loss, V_loss, loss_info = self.flow_energy_model.direct_flow2result_loss(
            observations=observations, actions=actions, next_observations=next_observations, rewards=rewards, dones=dones, tau=self.argus.iql_tau)
        self.q_optimizer.zero_grad()
        Q_loss.backward()
        self.q_optimizer.step()
        self.v_optimizer.zero_grad()
        V_loss.backward()
        self.v_optimizer.step()
        self.flow_energy_model.q_ema()
        return loss_info

    def direct_flow2result_policy_train(self, observations, actions):
        from models.rl_flow_forward_process import sample_weighted_interpolated_points

        num_samples = len(observations)
        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        pred_u = pred_u.clip(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        behavior_divergence = torch.norm(pred_u - behavior_u, dim=-1, keepdim=True)
        dt = 1.0 / self.argus.flow_step
        x_t = x_t + pred_u * dt
        next_pred_u = self.model(x=torch.cat([observations, x_t.detach()], dim=-1), t=t + dt)
        direction_consistency = torch.norm(
            pred_u / torch.norm(pred_u, dim=-1, keepdim=True) - next_pred_u / torch.norm(next_pred_u, dim=-1, keepdim=True),
            dim=-1, keepdim=True)
        x = torch.randn(num_samples, self.argus.action_dim, device=self.argus.device).clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        time_start, time_end, steps = 0, 1.0, self.argus.direct_flow_step
        for current_t in np.linspace(time_start, time_end, steps):
            t_tensor = torch.full((num_samples, 1), current_t, dtype=torch.float32, device=self.argus.device)
            direct_u = self.model(x=torch.cat([observations, x], dim=-1), t=t_tensor)
            x = x + direct_u * 1 / self.argus.direct_flow_step
        x = x.clamp(-self.argus.max_action_val, self.argus.max_action_val)
        pred_Q = self.flow_energy_model.get_scaled_q(obs=observations, act=x, scale=1.0)
        loss = -pred_Q + behavior_divergence + direction_consistency
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.q_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": behavior_divergence.mean().detach().cpu().numpy().item(),
            "direction_consistency": direction_consistency.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def adv_based_value_train(self, observations, actions):
        return adv_value_update(
            batch={
                "observations": observations,
                "actions": actions,
            },
            models={
                "energy_model": self.energy_model,
                "flow_energy_model": self.flow_energy_model,
            },
            optimizers={
                "fv_optimizer": self.fv_optimizer,
                "fv_v_optimizer": self.fv_v_optimizer,
            },
            config=self.argus,
        )

    def adv_based_policy_train(self, observations, actions):
        return adv_policy_update(
            batch={
                "observations": observations,
                "actions": actions,
            },
            models={
                "flow_model": self.model,
                "target_flow_model": self.target_model,
                "energy_model": self.energy_model,
                "flow_energy_model": self.flow_energy_model,
            },
            optimizers={
                "flow_optimizer": self.optimizer,
                "fv_optimizer": self.fv_optimizer,
            },
            config=self.argus,
        )

    def grpo_value_train(self, observations, actions):
        return self.adv_based_value_train(observations=observations, actions=actions)

    def grpo_policy_train(self, observations, actions, next_observations, next_actions, expectile=0.5):
        from models.rl_flow_forward_process import sample_weighted_interpolated_points

        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
            behavior_u = behavior_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            normed_behavior_u = behavior_u / torch.norm(behavior_u, dim=-1, keepdim=True)
            behavior_u_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_behavior_u], dim=-1), t=t)
            target_pred_u = self.target_model(x=torch.cat([observations, x_t], dim=-1), t=t)
            target_pred_u = target_pred_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            normed_target_pred_u = target_pred_u / torch.norm(target_pred_u, dim=-1, keepdim=True)
            target_pred_u_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_target_pred_u], dim=-1), t=t)
            Q_bar = torch.max(expectile * behavior_u_V + (1 - expectile) * target_pred_u_V, expectile * target_pred_u_V + (1 - expectile) * behavior_u_V)

        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        pred_u = pred_u.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
        divergence = self.loss_fn(pred_u, dx_dt)
        pred_Q = self.flow_energy_model.q(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        adv = self._normalize_advantage(pred_Q - Q_bar.detach())
        loss = -adv
        loss = self.argus.divergence_coef * divergence + loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def full_rl_like_value_train(self, observations, actions, next_observations, rewards, dones):
        from models.rl_flow_forward_process import sample_weighted_interpolated_points

        x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True)
        flow_observations = torch.cat([observations, x_t], dim=-1)
        flow_next_observations = torch.cat([next_observations, x_t], dim=-1)
        Q_loss, V_loss, loss_info = self.flow_energy_model.loss(
            tau=self.argus.iql_tau, observations=flow_observations, actions=dx_dt, normed_actions=normed_dx_dt,
            next_observations=flow_next_observations, t=t, next_actions=None, rewards=rewards, dones=dones,
            fake_actions=None, fake_next_actions=None)
        V_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow_energy_model.v.parameters(), max_norm=10.0)
        self.fv_v_optimizer.step()
        self.fv_v_optimizer.zero_grad()
        Q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow_energy_model.q.parameters(), max_norm=10.0)
        self.fv_optimizer.step()
        self.fv_optimizer.zero_grad()
        self.flow_energy_model.q_ema()
        return loss_info

    def full_rl_like_policy_train(self, observations, actions):
        from models.rl_flow_forward_process import sample_weighted_interpolated_points

        x_t, t, dx_dt, weights = sample_weighted_interpolated_points(
            argus=self.argus, observations=observations, actions=actions, energy=None, beta=self.argus.beta,
            energy_model=self.energy_model, weighted_samples_type=WeightedSamplesType.linear_interpolation,
            clip_value=self.argus.x_t_clip_value,
        )
        pred_u = self.model(x=torch.cat([observations, x_t], dim=-1), t=t)
        with torch.no_grad():
            behavior_u = self.behavior_flow(x=torch.cat([observations, x_t], dim=-1), t=t)
        cos_sim = torch.nn.functional.cosine_similarity(pred_u, behavior_u.detach(), dim=-1)
        divergence = 1 - cos_sim.unsqueeze(dim=-1)
        normed_pred_u = pred_u / torch.norm(pred_u, dim=-1, keepdim=True)
        pred_Q = self.flow_energy_model.q_target(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
        pred_V = self.flow_energy_model.v(x=torch.cat([observations, x_t, normed_pred_u], dim=-1), t=t)
        loss = -(pred_Q - pred_V)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.fv_optimizer.zero_grad()
        self.ema.update_model_average(self.target_model, self.model)
        return {
            "flow_divergence_coef": self.argus.divergence_coef,
            "flow_divergence": divergence.mean().detach().cpu().numpy().item(),
            "flow_loss": loss.detach().cpu().numpy().item(),
        }

    def save_critic_checpoint(self):
        data = {}
        if self.argus.critic_type in [CriticType.iql, CriticType.ciql]:
            data.update({
                'step': self.step,
                'q': self.energy_model.q.state_dict(),
                'q_target': self.energy_model.q_target.state_dict(),
                'v': self.energy_model.v.state_dict(),
                'behavior_flow': self.behavior_flow.state_dict(),
            })
        elif self.argus.critic_type in [CriticType.isql]:
            data.update({
                'step': self.step,
                'q': self.energy_model.q.state_dict(),
                'q_target': self.energy_model.q_target.state_dict(),
                'behavior_flow': self.behavior_flow.state_dict(),
            })
        else:
            raise ValueError(f"Critic type {self.argus.critic_type} not supported")
        savepath = os.path.join(self.get_detailed_save_path(), 'critic_checkpoint')
        os.makedirs(savepath, exist_ok=True)
        torch.save(data, os.path.join(savepath, f'critic_{self.step}.pt'))
        torch.save(data, os.path.join(savepath, 'critic.pt'))

    def get_checkpoint_step(self, loadpath, step_offset):
        checkpoints_list = os.listdir(loadpath)
        checkpoints_step = []
        for checkpoint in checkpoints_list:
            if "_" in checkpoint:
                checkpoints_step.append(int(checkpoint.split("_")[-1].split(".")[0]))
        checkpoints_step.sort()
        if step_offset == "latest":
            return checkpoints_step[-1]
        return checkpoints_step[0] + step_offset

    def load_critic_checkpoint(self, loadpath, step_offset="latest"):
        step = self.get_checkpoint_step(
            loadpath=os.path.join(loadpath, 'critic_checkpoint'), step_offset=step_offset)
        loadpath = os.path.join(loadpath, f'critic_checkpoint/critic_{step}.pt')
        data = torch.load(loadpath)
        if self.argus.critic_type in [CriticType.iql, CriticType.ciql]:
            self.energy_model.q.load_state_dict(data['q'])
            self.energy_model.q_target.load_state_dict(data['q_target'])
            self.energy_model.v.load_state_dict(data['v'])
            self.behavior_flow.load_state_dict(data['behavior_flow'])
        elif self.argus.critic_type in [CriticType.isql]:
            self.energy_model.q.load_state_dict(data['q'])
            self.energy_model.q_target.load_state_dict(data['q_target'])
            self.behavior_flow.load_state_dict(data['behavior_flow'])
        else:
            raise ValueError(f"Critic type {self.argus.critic_type} not supported")
