#!/usr/bin/env bash
set -euo pipefail

# Small deploy-behavior comparison.
# PRESET=delayed: train flow updates happen, but deployment waits briefly.
# PRESET=gradual: deployment starts early and ramps train-flow probability.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-Pendulum-v1}"
DOMAIN="${DOMAIN:-gymnasium}"
SEED="${SEED:-2}"
DEVICE="${DEVICE:-cpu}"
WANDB_LOG="${WANDB_LOG:-False}"
PRESET="${PRESET:-delayed}"
RUN_TAG="${RUN_TAG:-deploy_${PRESET}_seed${SEED}}"
LABEL="${LABEL:-debug_online_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-./output/debug_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${LABEL}.log}"

# Use complete short episodes so replay sampling, train-flow updates, and deploy
# gates are all observable. For Mujoco, increase these values.
INIT_STEPS="${INIT_STEPS:-208}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-208}"

mkdir -p "$LOG_DIR"

case "$PRESET" in
  delayed)
    DEPLOY_AFTER_UPDATES="${DEPLOY_AFTER_UPDATES:-4}"
    DEPLOY_AFTER_EPOCH="${DEPLOY_AFTER_EPOCH:-1}"
    GRADUAL_ENABLE="${GRADUAL_ENABLE:-False}"
    GRADUAL_START_PROB="${GRADUAL_START_PROB:-0.1}"
    GRADUAL_END_PROB="${GRADUAL_END_PROB:-1.0}"
    GRADUAL_RAMP_UPDATES="${GRADUAL_RAMP_UPDATES:-8}"
    ;;
  gradual)
    DEPLOY_AFTER_UPDATES="${DEPLOY_AFTER_UPDATES:-1}"
    DEPLOY_AFTER_EPOCH="${DEPLOY_AFTER_EPOCH:-0}"
    GRADUAL_ENABLE="${GRADUAL_ENABLE:-True}"
    GRADUAL_START_PROB="${GRADUAL_START_PROB:-0.2}"
    GRADUAL_END_PROB="${GRADUAL_END_PROB:-0.8}"
    GRADUAL_RAMP_UPDATES="${GRADUAL_RAMP_UPDATES:-8}"
    ;;
  *)
    echo "Unknown PRESET='$PRESET'. Use PRESET=delayed or PRESET=gradual." >&2
    exit 1
    ;;
esac

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
  --online_updates_per_epoch 6 \
  --update_flow_start_epoch 0 \
  --online_behavior_bootstrap_updates 0 \
  --online_eval_freq 6 \
  --eval_episodes 1 \
  --save_freq 100000 \
  --wandb_log_frequency 1 \
  --online_print_frequency 1 \
  --preserve_ep 30 \
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
  --online_min_flow_updates_before_deploy 1 \
  --online_deploy_flow_after_updates "$DEPLOY_AFTER_UPDATES" \
  --online_deploy_flow_after_epoch "$DEPLOY_AFTER_EPOCH" \
  --online_gradual_deploy_enable "$GRADUAL_ENABLE" \
  --online_gradual_deploy_start_prob "$GRADUAL_START_PROB" \
  --online_gradual_deploy_end_prob "$GRADUAL_END_PROB" \
  --online_gradual_deploy_ramp_updates "$GRADUAL_RAMP_UPDATES"
} 2>&1 | tee "$LOG_FILE"

if grep -Eiq '(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$LOG_FILE"; then
  echo "NaN detected in $LOG_FILE" >&2
  exit 1
fi
