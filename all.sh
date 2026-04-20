#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-HalfCheetah-v5}"
DOMAIN="${DOMAIN:-gymnasium}"
CRITIC_TYPE="${CRITIC_TYPE:-iql}"
RUN_TAG="${RUN_TAG:-swdg_w4_s1_n8}"
SEED="${SEED:-1}"
MAIN_GPU_POOL="${MAIN_GPU_POOL:-0 1}"
BASELINE_GPU_POOL="${BASELINE_GPU_POOL:-2 3}"

ONLINE_EPOCHS="${ONLINE_EPOCHS:-1000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1000}"
UPDATE_STEPS="${UPDATE_STEPS:-1000}"
ONLINE_UPDATE_MODE="${ONLINE_UPDATE_MODE:-epoch}"
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
BATCH_SIZE="${BATCH_SIZE:-512}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2}"
LR="${LR:-0.00005}"
DIVERGENCE_COEF="${DIVERGENCE_COEF:-3.0}"
ADV_BATCH_NORM="${ADV_BATCH_NORM:-True}"
CRITIC_UPDATE_INTERVAL="${CRITIC_UPDATE_INTERVAL:-1}"
PRESERVE_EP="${PRESERVE_EP:-300}"
MULTI_MODE_ACTION_EVALUATION="${MULTI_MODE_ACTION_EVALUATION:-False}"
ONLINE_ACTION_NOISE_ENABLE="${ONLINE_ACTION_NOISE_ENABLE:-True}"
ONLINE_ACTION_NOISE_STD="${ONLINE_ACTION_NOISE_STD:-0.5}"
ONLINE_ACTION_NOISE_CLIP="${ONLINE_ACTION_NOISE_CLIP:-1.0}"
ONLINE_ACTION_NOISE_DECAY_ENABLE="${ONLINE_ACTION_NOISE_DECAY_ENABLE:-True}"
ONLINE_ACTION_NOISE_START_STD="${ONLINE_ACTION_NOISE_START_STD:-0.5}"
ONLINE_ACTION_NOISE_END_STD="${ONLINE_ACTION_NOISE_END_STD:-0.1}"
ONLINE_ACTION_NOISE_DECAY_STEPS="${ONLINE_ACTION_NOISE_DECAY_STEPS:-1000000}"
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

validate_gpu_pool() {
  local pool_name="$1"
  local pool="$2"
  local -a gpus=()
  read -r -a gpus <<< "$pool"
  if (( ${#gpus[@]} == 0 )); then
    echo "${pool_name} must contain at least one GPU id." >&2
    exit 1
  fi
}

validate_gpu_pools_do_not_overlap() {
  local main_gpu
  local baseline_gpu
  for main_gpu in $MAIN_GPU_POOL; do
    for baseline_gpu in $BASELINE_GPU_POOL; do
      if [[ "$main_gpu" == "$baseline_gpu" ]]; then
        echo "MAIN_GPU_POOL and BASELINE_GPU_POOL overlap on GPU ${main_gpu}; refusing to oversubscribe." >&2
        exit 1
      fi
    done
  done
}

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
    --online_update_mode "$ONLINE_UPDATE_MODE" \
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
    --adv_batch_norm "$ADV_BATCH_NORM" \
    --online_critic_update_interval "$CRITIC_UPDATE_INTERVAL" \
    --preserve_ep "$PRESERVE_EP" \
    --multi_mode_action_evaluation "$MULTI_MODE_ACTION_EVALUATION" \
    --online_action_noise_enable "$ONLINE_ACTION_NOISE_ENABLE" \
    --online_action_noise_std "$ONLINE_ACTION_NOISE_STD" \
    --online_action_noise_clip "$ONLINE_ACTION_NOISE_CLIP" \
    --online_action_noise_decay_enable "$ONLINE_ACTION_NOISE_DECAY_ENABLE" \
    --online_action_noise_start_std "$ONLINE_ACTION_NOISE_START_STD" \
    --online_action_noise_end_std "$ONLINE_ACTION_NOISE_END_STD" \
    --online_action_noise_decay_steps "$ONLINE_ACTION_NOISE_DECAY_STEPS" \
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

wait_for_free_gpu() {
  local -n wait_pids_ref="$1"
  local idx
  while true; do
    for idx in "${!wait_pids_ref[@]}"; do
      if [[ -z "${wait_pids_ref[$idx]}" ]]; then
        FREE_GPU_IDX="$idx"
        return 0
      fi
      if ! kill -0 "${wait_pids_ref[$idx]}" 2>/dev/null; then
        wait "${wait_pids_ref[$idx]}"
        wait_pids_ref[$idx]=""
        FREE_GPU_IDX="$idx"
        return 0
      fi
    done
    sleep 10
  done
}

schedule_experiment_group() {
  local group="$1"
  local pool="$2"
  shift 2
  local -a gpus=()
  local -a pids=()
  local spec
  local gpu_idx
  local gpu
  local label
  local seed
  local behavior_only
  local FREE_GPU_IDX

  read -r -a gpus <<< "$pool"
  for gpu_idx in "${!gpus[@]}"; do
    pids[$gpu_idx]=""
  done

  for spec in "$@"; do
    IFS='|' read -r label seed behavior_only <<< "$spec"
    wait_for_free_gpu pids
    gpu_idx="$FREE_GPU_IDX"
    gpu="${gpus[$gpu_idx]}"
    echo "============================================================"
    echo "Launching group=${group} gpu=${gpu} experiment=${label}_seed${seed}_${RUN_TAG}"
    echo "============================================================"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      run_experiment "$label" "$seed" "$behavior_only"
    ) &
    pids[$gpu_idx]="$!"
  done

  for gpu_idx in "${!pids[@]}"; do
    if [[ -n "${pids[$gpu_idx]}" ]]; then
      wait "${pids[$gpu_idx]}"
      pids[$gpu_idx]=""
    fi
  done
}

validate_gpu_pool "MAIN_GPU_POOL" "$MAIN_GPU_POOL"
validate_gpu_pool "BASELINE_GPU_POOL" "$BASELINE_GPU_POOL"
validate_gpu_pools_do_not_overlap

main_experiments=(
  "adv_rl_swdg|$SEED|False"
)

baseline_experiments=(
  "behavior_only_swdg|$SEED|True"
)

schedule_experiment_group "main" "$MAIN_GPU_POOL" "${main_experiments[@]}" &
main_scheduler_pid="$!"
schedule_experiment_group "baseline" "$BASELINE_GPU_POOL" "${baseline_experiments[@]}" &
baseline_scheduler_pid="$!"

main_status=0
baseline_status=0
wait "$main_scheduler_pid" || main_status="$?"
wait "$baseline_scheduler_pid" || baseline_status="$?"
if (( main_status != 0 || baseline_status != 0 )); then
  exit 1
fi
