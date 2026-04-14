# UV Setup

This project now targets Python 3.10 for the modern online Mujoco pipeline.

## Create the environment

```bash
uv python install 3.10
uv venv --python 3.10
```

## Install the online training stack

```bash
uv sync
```

## Optional legacy offline stack

Only install this if you still need D4RL and the old `mujoco-py` pipeline.

```bash
uv sync --extra offline-d4rl
```

## Run the online experiment

```bash
uv run python multi_step_online_rl_flow_main.py train \
  --dataset HalfCheetah-v5 \
  --domain gymnasium \
  --rl_mode grpo \
  --critic_type iql \
  --current_exp_label online_uv
```

## W&B login

```bash
uv run wandb login
```
