#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-1}"
DATASET="${DATASET:-HalfCheetah-v5}"
DOMAIN="${DOMAIN:-gymnasium}"
RUN_TAG="${RUN_TAG:-debug_flow_gate}"

COMMON_ARGS=(
  --dataset "$DATASET"
  --domain "$DOMAIN"
  --rl_mode adv_rl
  --critic_type iql
  --seed "$SEED"
  --reset_seed False
  --online_behavior_only False
  --online_use_sliding_window_critic True
  --swdg_num_q_ensembles 8
  --swdg_window_size 4
  --swdg_window_step 1
  --online_epochs "${ONLINE_EPOCHS:-20}"
  --online_rollout_steps_per_epoch "${ROLLOUT_STEPS:-1000}"
  --online_updates_per_epoch "${UPDATE_STEPS:-300}"
  --online_update_mode "${ONLINE_UPDATE_MODE:-epoch}"
  --online_init_steps "${INIT_STEPS:-5000}"
  --online_random_steps "${RANDOM_STEPS:-5000}"
  --online_eval_freq "${EVAL_FREQ:-1000}"
  --wandb_log_frequency "${BATCH_LOG_FREQ:-100}"
  --online_print_frequency "${PRINT_FREQ:-200}"
  --batch_size "${BATCH_SIZE:-256}"
  --sequence_length 2
  --lr "${LR:-0.00005}"
  --divergence_coef "${DIVERGENCE_COEF:-3.0}"
  --preserve_ep "${PRESERVE_EP:-300}"
  --online_adv_batch_norm True
  --multi_mode_action_evaluation False
  --online_action_noise_enable True
  --online_action_noise_std "${ONLINE_ACTION_NOISE_STD:-0.3}"
  --online_action_noise_clip "${ONLINE_ACTION_NOISE_CLIP:-1.0}"
  --online_behavior_bootstrap_updates "${ONLINE_BEHAVIOR_BOOTSTRAP_UPDATES:-300}"
  --online_init_train_flow_from_behavior True
)

echo "============================================================"
echo "Debug A: deploy behavior_flow only by setting a huge update_flow_start_epoch"
echo "============================================================"
uv run python multi_step_online_rl_flow_main.py train \
  "${COMMON_ARGS[@]}" \
  --current_exp_label "debug_no_trainflow_deploy_seed${SEED}_${RUN_TAG}" \
  --update_flow_start_epoch 100000 \
  --online_min_flow_updates_before_deploy 1

echo "============================================================"
echo "Debug B: enable normal adv_rl flow updates with sequence_length=2"
echo "============================================================"
uv run python multi_step_online_rl_flow_main.py train \
  "${COMMON_ARGS[@]}" \
  --current_exp_label "debug_seq2_flow_updates_seed${SEED}_${RUN_TAG}" \
  --update_flow_start_epoch "${UPDATE_FLOW_START_EPOCH:-10}" \
  --online_min_flow_updates_before_deploy "${ONLINE_MIN_FLOW_UPDATES_BEFORE_DEPLOY:-1}"
