#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-HalfCheetah-v5}"
DOMAIN="${DOMAIN:-gymnasium}"
CRITIC_TYPE="${CRITIC_TYPE:-iql}"
RUN_TAG="${RUN_TAG:-swdg_w4_s1_n8_late_decay}"
SEED="${SEED:-1}"

ONLINE_EPOCHS="${ONLINE_EPOCHS:-100}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1000}"
UPDATE_STEPS="${UPDATE_STEPS:-300}"
INIT_STEPS="${INIT_STEPS:-1000}"
RANDOM_STEPS="${RANDOM_STEPS:-1000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
BATCH_LOG_FREQ="${BATCH_LOG_FREQ:-100}"
PRINT_FREQ="${PRINT_FREQ:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.00005}"
DIVERGENCE_COEF="${DIVERGENCE_COEF:-3.0}"
PRESERVE_EP="${PRESERVE_EP:-100}"
ONLINE_ADV_BATCH_NORM="${ONLINE_ADV_BATCH_NORM:-True}"
ONLINE_ACTION_NOISE_ENABLE="${ONLINE_ACTION_NOISE_ENABLE:-True}"
ONLINE_ACTION_NOISE_STD="${ONLINE_ACTION_NOISE_STD:-0.3}"
ONLINE_ACTION_NOISE_CLIP="${ONLINE_ACTION_NOISE_CLIP:-1.0}"
ONLINE_USE_SLIDING_WINDOW_CRITIC="${ONLINE_USE_SLIDING_WINDOW_CRITIC:-True}"
SWDG_NUM_Q_ENSEMBLES="${SWDG_NUM_Q_ENSEMBLES:-8}"
SWDG_WINDOW_SIZE="${SWDG_WINDOW_SIZE:-4}"
SWDG_WINDOW_STEP="${SWDG_WINDOW_STEP:-1}"
SWDG_USE_DIVERSITY_REG="${SWDG_USE_DIVERSITY_REG:-False}"
SWDG_DIVERSITY_COEF="${SWDG_DIVERSITY_COEF:-0.0}"
UPDATE_FLOW_START_EPOCH="${UPDATE_FLOW_START_EPOCH:-5}"
LATE_PHASE_START_EPOCH="${LATE_PHASE_START_EPOCH:-60}"

run_experiment() {
  local label="$1"
  local behavior_only="$2"
  local late_lr_scale="$3"
  local late_divergence_scale="$4"
  local late_update_ratio_scale="$5"

  echo "============================================================"
  echo "Running label=${label} seed=${SEED} behavior_only=${behavior_only}"
  echo "Late schedule: lr_scale=${late_lr_scale} div_scale=${late_divergence_scale} utd_scale=${late_update_ratio_scale}"
  echo "============================================================"

  uv run python multi_step_online_rl_flow_main.py train \
    --dataset "$DATASET" \
    --domain "$DOMAIN" \
    --rl_mode adv_rl \
    --critic_type "$CRITIC_TYPE" \
    --current_exp_label "${label}_seed${SEED}_${RUN_TAG}" \
    --seed "$SEED" \
    --reset_seed False \
    --online_behavior_only "$behavior_only" \
    --online_use_sliding_window_critic "$ONLINE_USE_SLIDING_WINDOW_CRITIC" \
    --swdg_num_q_ensembles "$SWDG_NUM_Q_ENSEMBLES" \
    --swdg_window_size "$SWDG_WINDOW_SIZE" \
    --swdg_window_step "$SWDG_WINDOW_STEP" \
    --swdg_use_diversity_reg "$SWDG_USE_DIVERSITY_REG" \
    --swdg_diversity_coef "$SWDG_DIVERSITY_COEF" \
    --online_epochs "$ONLINE_EPOCHS" \
    --online_rollout_steps_per_epoch "$ROLLOUT_STEPS" \
    --online_updates_per_epoch "$UPDATE_STEPS" \
    --online_init_steps "$INIT_STEPS" \
    --online_random_steps "$RANDOM_STEPS" \
    --update_flow_start_epoch "$UPDATE_FLOW_START_EPOCH" \
    --online_eval_freq "$EVAL_FREQ" \
    --wandb_log_frequency "$BATCH_LOG_FREQ" \
    --online_print_frequency "$PRINT_FREQ" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --divergence_coef "$DIVERGENCE_COEF" \
    --preserve_ep "$PRESERVE_EP" \
    --online_adv_batch_norm "$ONLINE_ADV_BATCH_NORM" \
    --online_action_noise_enable "$ONLINE_ACTION_NOISE_ENABLE" \
    --online_action_noise_std "$ONLINE_ACTION_NOISE_STD" \
    --online_action_noise_clip "$ONLINE_ACTION_NOISE_CLIP" \
    --online_late_phase_start_epoch "$LATE_PHASE_START_EPOCH" \
    --online_late_lr_scale "$late_lr_scale" \
    --online_late_divergence_scale "$late_divergence_scale" \
    --online_late_update_ratio_scale "$late_update_ratio_scale"
}

run_experiment "adv_rl_swdg_late_lr" "False" "0.5" "1.0" "1.0"
run_experiment "adv_rl_swdg_late_div" "False" "1.0" "0.5" "1.0"
run_experiment "adv_rl_swdg_late_utd" "False" "1.0" "1.0" "0.5"
