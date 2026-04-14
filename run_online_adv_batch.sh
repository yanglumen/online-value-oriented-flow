#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-HalfCheetah-v5}"
DOMAIN="${DOMAIN:-gymnasium}"
CRITIC_TYPE="${CRITIC_TYPE:-iql}"
ADV_SEEDS=(${ADV_SEEDS:-1 2})

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

run_experiment() {
  local label="$1"
  local seed="$2"
  local behavior_only="$3"

  echo "============================================================"
  echo "Running label=${label} seed=${seed} behavior_only=${behavior_only}"
  echo "============================================================"

  uv run python multi_step_online_rl_flow_main.py train \
    --dataset "$DATASET" \
    --domain "$DOMAIN" \
    --rl_mode adv_rl \
    --critic_type "$CRITIC_TYPE" \
    --current_exp_label "${label}_seed${seed}" \
    --seed "$seed" \
    --reset_seed False \
    --online_behavior_only "$behavior_only" \
    --online_epochs "$ONLINE_EPOCHS" \
    --online_rollout_steps_per_epoch "$ROLLOUT_STEPS" \
    --online_updates_per_epoch "$UPDATE_STEPS" \
    --online_init_steps "$INIT_STEPS" \
    --online_random_steps "$RANDOM_STEPS" \
    --online_eval_freq "$EVAL_FREQ" \
    --wandb_log_frequency "$BATCH_LOG_FREQ" \
    --online_print_frequency "$PRINT_FREQ" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --divergence_coef "$DIVERGENCE_COEF" \
    --preserve_ep "$PRESERVE_EP" \
    --online_adv_batch_norm "$ONLINE_ADV_BATCH_NORM"
}

for seed in "${ADV_SEEDS[@]}"; do
  run_experiment "adv_rl_lr5e5_bn_recentbuf" "$seed" "False"
done
