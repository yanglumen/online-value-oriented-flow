# AGENTS.md

This file gives Codex project-specific instructions for the `value-oriented-flow` repository.

## Scope

These instructions apply to the repository root and all subdirectories unless a deeper `AGENTS.md` overrides them.

## Project mission

This repo contains research code for value-oriented flow matching in reinforcement learning.

The highest-priority path is the modern online Mujoco pipeline based on Gymnasium:
- canonical online entrypoint: `multi_step_online_rl_flow_main.py`
- canonical online trainer: `trainer/multi_step_online_rl_flow_trainer.py`
- shared online RL update core: `trainer/shared_flow_rl_core.py`

There are also older offline / D4RL-style experiment entrypoints. Treat them as legacy unless the task explicitly targets them.

## Environment and commands

Use Python 3.10.

Preferred setup:
- `uv sync`
- optional legacy stack: `uv sync --extra offline-d4rl`

Preferred execution style:
- `uv run python ...`
- `uv run wandb login` if W&B is needed

Do not introduce new dependencies unless the task explicitly requires them.

## Repository map

### Main entrypoints
- `multi_step_online_rl_flow_main.py`: main online training entrypoint for Gymnasium Mujoco.
- `multistep_rl_flow_main.py`: older offline / dataset-driven RL-flow entrypoint.
- `pretrain_value_models.py`: value pretraining utilities.
- other `*_main.py` files are experiment variants; do not refactor them casually.

### Core training code
- `trainer/multi_step_online_rl_flow_trainer.py`: online rollout loop, replay usage, deploy gates, eval cadence, checkpointing.
- `trainer/shared_flow_rl_core.py`: behavior-flow update, critic update, flow-value update, flow-policy update.
- `trainer/trainer_util.py`: seeding and helper utilities.

### Core models
- `models/flow_model.py`: behavior and train flow networks.
- `models/flow_value_model.py`: flow critic used by adv-style flow policy training.
- `models/sliding_window_iql_critic.py`: online teacher critic for the current online setup.
- `models/energy_model.py`: older energy / critic implementations.

### Config and data
- `config/multistep_rl_flow_hyperparameter.py`: enums and default hyperparameters.
- `datasets_process/online_sequence_dataset.py`: online replay / sampling path for the Gymnasium pipeline.
- `evaluation/`: evaluation helpers.

### Useful scripts
- `all.sh`: current primary entrypoint for the user's main training workflow; treat this as the default launcher unless the task explicitly targets another script.
- `debug_online_smoke.sh`: the safest quick validation path after online changes.
- `all_debug_flow_gate.sh`: targeted debugging for train-flow deployment / gating behavior.
- `run_online_all.sh`: multi-run launcher used for normal online sweeps.

## Workflow rules

1. Prefer minimal, local changes.
   - This is research code with many experiment entrypoints.
   - Avoid broad renames or architectural cleanup unless explicitly requested.

2. Preserve experiment semantics.
   - Do not silently change rollout/update ratios, replay semantics, eval policy, deploy thresholds, or normalization behavior.
   - If a task requires changing any of these, make the change explicit and mention the semantic effect.

3. Preserve CLI compatibility where practical.
   - Many scripts depend on Fire CLI arguments and config keys.
   - If you rename or remove an argument, update all affected launch scripts and validation logic.
   - Pay special attention to `import_parameters(...)` guardrails in entrypoint files.

4. Treat online and offline pipelines separately.
   - The online path is Gymnasium-based and Python 3.10 friendly.
   - The legacy offline path may still depend on older D4RL / gym / mujoco-py assumptions.
   - Do not mix APIs between them unless the task is specifically about migration.

5. Respect the current research decomposition.
   - `behavior_flow`, `train_flow`, `flow_energy_model`, and `energy_model` have distinct roles.
   - When changing update logic, inspect both the trainer and the shared RL core.

6. Prefer explicit diagnostics over hidden behavior.
   - If a bug is related to training schedule, logging, deployment gates, or critic update cadence, add or preserve counters / metrics.
   - Do not remove debugging logs lightly from the online trainer.

## When editing online training logic

If the task touches online RL behavior, inspect these files first:
- `multi_step_online_rl_flow_main.py`
- `trainer/multi_step_online_rl_flow_trainer.py`
- `trainer/shared_flow_rl_core.py`
- `models/sliding_window_iql_critic.py`
- `models/flow_value_model.py`
- `datasets_process/online_sequence_dataset.py`

Important patterns in this repo:
- Hyperparameter defaults may be defined in config but constrained again inside the online entrypoint.
- Launch scripts may pass arguments that are later transformed, validated, or rejected.
- Seemingly small changes to update cadence can alter experimental semantics significantly.

## Validation expectations

Use the lightest validation that matches the scope of the change.

### For small Python-only edits
Run syntax checks on touched files, for example:
- `uv run python -m py_compile multi_step_online_rl_flow_main.py trainer/*.py models/*.py config/*.py`

### For online training changes
Run the smoke test first:
- `bash debug_online_smoke.sh`

If the change touches deployment gating, actor/critic cadence, or rollout/update scheduling, also consider:
- `bash all_debug_flow_gate.sh`

### For longer online runs
Prefer a reduced local run before large sweeps, for example a 1-epoch or low-step command using `multi_step_online_rl_flow_main.py train`.

## Coding style

- Match the existing PyTorch style and naming conventions.
- Keep changes readable and explicit rather than clever.
- Avoid introducing framework-wide abstractions unless asked.
- Keep comments high-signal and task-specific.
- Preserve existing metric names when possible because scripts and dashboards may rely on them.

## Output and artifacts

- Training outputs are usually written under `./output`.
- Do not commit large generated artifacts, checkpoints, logs, or W&B cache files unless the task explicitly asks for them.

## Safe defaults for Codex

When the user asks for help without specifying the pipeline, assume they mean the online Gymnasium pipeline.

When a task is ambiguous between “refactor” and “debug an experiment,” prefer preserving behavior and adding the smallest possible fix.

When reporting results, include:
- which files were changed
- what experimental semantics changed, if any
- what validation was run
