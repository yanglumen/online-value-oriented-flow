#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-HalfCheetah-v5}"
DOMAIN="${DOMAIN:-gymnasium}"
CRITIC_TYPE="${CRITIC_TYPE:-iql}"
RUN_TAG="${RUN_TAG:-swdg_w4_s1_n8}"
SEED="${SEED:-1}"

ONLINE_EPOCHS="${ONLINE_EPOCHS:-1000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1000}"
UPDATE_STEPS="${UPDATE_STEPS:-300}"
INIT_STEPS="${INIT_STEPS:-5000}"
RANDOM_STEPS="${RANDOM_STEPS:-5000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
BATCH_LOG_FREQ="${BATCH_LOG_FREQ:-100}"
PRINT_FREQ="${PRINT_FREQ:-200}"
WANDB_LOG="${WANDB_LOG:-True}"
# Leave empty for normal online W&B logging. Set WANDB_MODE=offline only when
# intentionally writing local runs that will be synced later.
WANDB_MODE="${WANDB_MODE:-}"
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2}"
LR="${LR:-0.00005}"
DIVERGENCE_COEF="${DIVERGENCE_COEF:-3.0}"
PRESERVE_EP="${PRESERVE_EP:-300}"
MULTI_MODE_ACTION_EVALUATION="${MULTI_MODE_ACTION_EVALUATION:-False}"
ONLINE_ACTION_NOISE_ENABLE="${ONLINE_ACTION_NOISE_ENABLE:-True}"
ONLINE_ACTION_NOISE_STD="${ONLINE_ACTION_NOISE_STD:-0.3}"
ONLINE_ACTION_NOISE_CLIP="${ONLINE_ACTION_NOISE_CLIP:-1.0}"
ONLINE_EVAL_DETERMINISTIC="${ONLINE_EVAL_DETERMINISTIC:-True}"
ONLINE_EVAL_STOCHASTIC="${ONLINE_EVAL_STOCHASTIC:-True}"
ONLINE_USE_SLIDING_WINDOW_CRITIC="${ONLINE_USE_SLIDING_WINDOW_CRITIC:-True}"
SWDG_NUM_Q_ENSEMBLES="${SWDG_NUM_Q_ENSEMBLES:-8}"
SWDG_WINDOW_SIZE="${SWDG_WINDOW_SIZE:-4}"
SWDG_WINDOW_STEP="${SWDG_WINDOW_STEP:-1}"
SWDG_USE_DIVERSITY_REG="${SWDG_USE_DIVERSITY_REG:-False}"
SWDG_DIVERSITY_COEF="${SWDG_DIVERSITY_COEF:-0.0}"
UPDATE_FLOW_START_EPOCH="${UPDATE_FLOW_START_EPOCH:-10}"
ONLINE_BEHAVIOR_BOOTSTRAP_UPDATES="${ONLINE_BEHAVIOR_BOOTSTRAP_UPDATES:-$UPDATE_STEPS}"
ONLINE_MIN_FLOW_UPDATES_BEFORE_DEPLOY="${ONLINE_MIN_FLOW_UPDATES_BEFORE_DEPLOY:-1}"
ONLINE_INIT_TRAIN_FLOW_FROM_BEHAVIOR="${ONLINE_INIT_TRAIN_FLOW_FROM_BEHAVIOR:-True}"
ONLINE_DEPLOY_FLOW_AFTER_UPDATES="${ONLINE_DEPLOY_FLOW_AFTER_UPDATES:-3000}"
ONLINE_DEPLOY_FLOW_AFTER_EPOCH="${ONLINE_DEPLOY_FLOW_AFTER_EPOCH:-20}"
ONLINE_GRADUAL_DEPLOY_ENABLE="${ONLINE_GRADUAL_DEPLOY_ENABLE:-False}"
ONLINE_GRADUAL_DEPLOY_START_PROB="${ONLINE_GRADUAL_DEPLOY_START_PROB:-0.1}"
ONLINE_GRADUAL_DEPLOY_END_PROB="${ONLINE_GRADUAL_DEPLOY_END_PROB:-1.0}"
ONLINE_GRADUAL_DEPLOY_RAMP_UPDATES="${ONLINE_GRADUAL_DEPLOY_RAMP_UPDATES:-5000}"

if (( SEQUENCE_LENGTH < 2 )); then
  echo "SEQUENCE_LENGTH must be >= 2 for online adv_rl training." >&2
  exit 1
fi

run_experiment() {
  local label="$1"
  local seed="$2"
  local behavior_only="$3"

  echo "============================================================"
  echo "Running label=${label} seed=${seed} behavior_only=${behavior_only}"
  echo "============================================================"

  local wandb_mode_args=()
  if [[ -n "$WANDB_MODE" ]]; then
    wandb_mode_args=(--wandb_mode "$WANDB_MODE")
  fi

  uv run python multi_step_online_rl_flow_main.py train \
    --dataset "$DATASET" \
    --domain "$DOMAIN" \
    --rl_mode adv_rl \
    --critic_type "$CRITIC_TYPE" \
    --current_exp_label "${label}_seed${seed}_${RUN_TAG}" \
    --seed "$seed" \
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
    --wandb_log "$WANDB_LOG" \
    "${wandb_mode_args[@]}" \
    --wandb_init_timeout "$WANDB_INIT_TIMEOUT" \
    --wandb_log_frequency "$BATCH_LOG_FREQ" \
    --online_print_frequency "$PRINT_FREQ" \
    --batch_size "$BATCH_SIZE" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --lr "$LR" \
    --divergence_coef "$DIVERGENCE_COEF" \
    --preserve_ep "$PRESERVE_EP" \
    --multi_mode_action_evaluation "$MULTI_MODE_ACTION_EVALUATION" \
    --online_action_noise_enable "$ONLINE_ACTION_NOISE_ENABLE" \
    --online_action_noise_std "$ONLINE_ACTION_NOISE_STD" \
    --online_action_noise_clip "$ONLINE_ACTION_NOISE_CLIP" \
    --online_eval_deterministic "$ONLINE_EVAL_DETERMINISTIC" \
    --online_eval_stochastic "$ONLINE_EVAL_STOCHASTIC" \
    --online_behavior_bootstrap_updates "$ONLINE_BEHAVIOR_BOOTSTRAP_UPDATES" \
    --online_min_flow_updates_before_deploy "$ONLINE_MIN_FLOW_UPDATES_BEFORE_DEPLOY" \
    --online_init_train_flow_from_behavior "$ONLINE_INIT_TRAIN_FLOW_FROM_BEHAVIOR" \
    --online_deploy_flow_after_updates "$ONLINE_DEPLOY_FLOW_AFTER_UPDATES" \
    --online_deploy_flow_after_epoch "$ONLINE_DEPLOY_FLOW_AFTER_EPOCH" \
    --online_gradual_deploy_enable "$ONLINE_GRADUAL_DEPLOY_ENABLE" \
    --online_gradual_deploy_start_prob "$ONLINE_GRADUAL_DEPLOY_START_PROB" \
    --online_gradual_deploy_end_prob "$ONLINE_GRADUAL_DEPLOY_END_PROB" \
    --online_gradual_deploy_ramp_updates "$ONLINE_GRADUAL_DEPLOY_RAMP_UPDATES"
}

run_experiment "behavior_only_swdg" "$SEED" "True"
run_experiment "adv_rl_swdg" "$SEED" "False"
