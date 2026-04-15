#!/usr/bin/env bash
set -euo pipefail

# Shadow-mode online debug run.
# The train_flow is updated, but deployment is disabled by impossible deploy
# thresholds. Use this to check losses, Q diagnostics, and eval stability without
# policy deployment confounds.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-Pendulum-v1}"
DOMAIN="${DOMAIN:-gymnasium}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cpu}"
WANDB_LOG="${WANDB_LOG:-False}"
RUN_TAG="${RUN_TAG:-shadow_seed${SEED}}"
LABEL="${LABEL:-debug_online_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-./output/debug_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${LABEL}.log}"

# Keep this long enough to finalize at least one Pendulum episode and cheap
# enough for quick loss/NaN checks. For Mujoco, increase both values.
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
  --batch_size 16 \
  --lr 0.00005 \
  --divergence_coef 3.0 \
  --normalizer OnlineGaussianNormalizer \
  --online_epochs 2 \
  --online_init_steps "$INIT_STEPS" \
  --online_random_steps "$INIT_STEPS" \
  --online_rollout_steps_per_epoch "$ROLLOUT_STEPS" \
  --online_updates_per_epoch 4 \
  --update_flow_start_epoch 0 \
  --online_behavior_bootstrap_updates 0 \
  --online_eval_freq 4 \
  --eval_episodes 1 \
  --save_freq 100000 \
  --wandb_log_frequency 1 \
  --online_print_frequency 1 \
  --preserve_ep 20 \
  --online_eval_deterministic True \
  --online_eval_stochastic True \
  --online_action_noise_enable True \
  --online_action_noise_std 0.1 \
  --online_action_noise_clip 0.3 \
  --online_use_sliding_window_critic True \
  --swdg_num_q_ensembles 4 \
  --swdg_window_size 2 \
  --swdg_window_step 1 \
  --swdg_use_diversity_reg False \
  --swdg_diversity_coef 0.0 \
  --online_init_train_flow_from_behavior True \
  --online_min_flow_updates_before_deploy 999999 \
  --online_deploy_flow_after_updates 999999 \
  --online_deploy_flow_after_epoch 999999 \
  --online_gradual_deploy_enable False
} 2>&1 | tee "$LOG_FILE"

if grep -Eiq '(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$LOG_FILE"; then
  echo "NaN detected in $LOG_FILE" >&2
  exit 1
fi
