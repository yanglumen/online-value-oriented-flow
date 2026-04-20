#!/usr/bin/env bash
set -euo pipefail

# Minimal online smoke test for the refactored shared training core.
# Verifies: env/model creation, replay warmup, one rollout cycle, a few updates,
# deterministic+stochastic eval, and checkpoint saving.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-Pendulum-v1}"
DOMAIN="${DOMAIN:-gymnasium}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cpu}"
WANDB_LOG="${WANDB_LOG:-False}"
RUN_TAG="${RUN_TAG:-smoke_seed${SEED}}"
LABEL="${LABEL:-debug_online_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-./output/debug_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${LABEL}.log}"

# Pendulum-v1 episodes are 200 steps. Use 208 so replay gets at least one
# finalized episode; override these when debugging long-horizon Mujoco envs.
INIT_STEPS="${INIT_STEPS:-208}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-208}"

mkdir -p "$LOG_DIR"

{
uv run python multi_step_online_rl_flow_main.py train \
  --dataset "$DATASET" \
  --domain "$DOMAIN" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --reset_seed False \
  --current_exp_label "$LABEL" \
  --wandb_log "$WANDB_LOG" \
  --rl_mode adv_rl \
  --critic_type iql \
  --sequence_length 2 \
  --batch_size 8 \
  --lr 0.00005 \
  --divergence_coef 3.0 \
  --normalizer OnlineGaussianNormalizer \
  --online_epochs 1 \
  --online_init_steps "$INIT_STEPS" \
  --online_random_steps "$INIT_STEPS" \
  --online_rollout_steps_per_epoch "$ROLLOUT_STEPS" \
  --online_updates_per_epoch 3 \
  --online_update_mode epoch \
  --update_flow_start_epoch 0 \
  --online_behavior_bootstrap_updates 0 \
  --online_eval_freq 1 \
  --eval_episodes 1 \
  --save_freq 1 \
  --wandb_log_frequency 1 \
  --online_print_frequency 1 \
  --preserve_ep 10 \
  --online_eval_deterministic True \
  --online_eval_stochastic True \
  --online_action_noise_enable False \
  --online_use_sliding_window_critic True \
  --swdg_num_q_ensembles 2 \
  --swdg_window_size 2 \
  --swdg_window_step 1 \
  --swdg_use_diversity_reg False \
  --swdg_diversity_coef 0.0 \
  --online_init_train_flow_from_behavior True \
  --online_min_flow_updates_before_deploy 1 \
  --online_deploy_flow_after_updates 9999 \
  --online_deploy_flow_after_epoch 9999
} 2>&1 | tee "$LOG_FILE"

if grep -Eiq '(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$LOG_FILE"; then
  echo "NaN detected in $LOG_FILE" >&2
  exit 1
fi
