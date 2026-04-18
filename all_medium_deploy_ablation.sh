#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-HalfCheetah-v5}"
DOMAIN="${DOMAIN:-gymnasium}"
SEEDS=(${SEEDS:-1 2 3})
RUN_TAG="${RUN_TAG:-seq2_deploy_ablation}"

ONLINE_EPOCHS="${ONLINE_EPOCHS:-200}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1000}"
UPDATE_STEPS="${UPDATE_STEPS:-$ROLLOUT_STEPS}"
INIT_STEPS="${INIT_STEPS:-5000}"
RANDOM_STEPS="${RANDOM_STEPS:-5000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
BATCH_LOG_FREQ="${BATCH_LOG_FREQ:-100}"
PRINT_FREQ="${PRINT_FREQ:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.00005}"
DIVERGENCE_COEF="${DIVERGENCE_COEF:-3.0}"
PRESERVE_EP="${PRESERVE_EP:-300}"
UPDATE_FLOW_START_EPOCH="${UPDATE_FLOW_START_EPOCH:-10}"
ONLINE_BEHAVIOR_BOOTSTRAP_UPDATES="${ONLINE_BEHAVIOR_BOOTSTRAP_UPDATES:-$UPDATE_STEPS}"

run_adv() {
  local label="$1"
  local seed="$2"
  local deploy_updates="$3"
  local deploy_epoch="$4"
  local gradual="$5"

  echo "============================================================"
  echo "Running ${label} seed=${seed} deploy_updates=${deploy_updates} deploy_epoch=${deploy_epoch} gradual=${gradual}"
  echo "============================================================"

  uv run python multi_step_online_rl_flow_main.py train \
    --dataset "$DATASET" \
    --domain "$DOMAIN" \
    --rl_mode adv_rl \
    --critic_type iql \
    --current_exp_label "${label}_seed${seed}_${RUN_TAG}" \
    --seed "$seed" \
    --reset_seed False \
    --online_behavior_only False \
    --online_use_sliding_window_critic True \
    --swdg_num_q_ensembles 8 \
    --swdg_window_size 4 \
    --swdg_window_step 1 \
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
    --sequence_length 2 \
    --lr "$LR" \
    --divergence_coef "$DIVERGENCE_COEF" \
    --preserve_ep "$PRESERVE_EP" \
    --online_adv_batch_norm True \
    --multi_mode_action_evaluation False \
    --online_action_noise_enable True \
    --online_action_noise_std 0.3 \
    --online_action_noise_clip 1.0 \
    --online_behavior_bootstrap_updates "$ONLINE_BEHAVIOR_BOOTSTRAP_UPDATES" \
    --online_min_flow_updates_before_deploy 1 \
    --online_init_train_flow_from_behavior True \
    --online_deploy_flow_after_updates "$deploy_updates" \
    --online_deploy_flow_after_epoch "$deploy_epoch" \
    --online_gradual_deploy_enable "$gradual" \
    --online_gradual_deploy_start_prob 0.1 \
    --online_gradual_deploy_end_prob 1.0 \
    --online_gradual_deploy_ramp_updates 5000
}

for seed in "${SEEDS[@]}"; do
  run_adv "shadow_never_deploy" "$seed" "100000000" "100000" "False"
  run_adv "delayed_deploy" "$seed" "3000" "30" "False"
  run_adv "gradual_deploy" "$seed" "3000" "30" "True"
done
