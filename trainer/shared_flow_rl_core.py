import torch
import torch.nn.functional as F

from config.multistep_rl_flow_hyperparameter import WeightedSamplesType
from models.rl_flow_forward_process import sample_weighted_interpolated_points


def _item(tensor):
    return tensor.detach().cpu().numpy().item()


def _ema_update(target_model, source_model, beta):
    for source_params, target_params in zip(source_model.parameters(), target_model.parameters()):
        target_params.data = target_params.data * beta + source_params.data * (1 - beta)


def generate_train_flow_action(flow_model, critic, states, previous_actions, config, deterministic=False, flow_steps=None):
    return flow_model.gen_action(
        states=states,
        critic=critic,
        executed_actions=previous_actions,
        steps=config.flow_step if flow_steps is None else flow_steps,
        x_t_clip_value=config.x_t_clip_value,
        deterministic=deterministic,
    ).clamp(-config.max_action_val, config.max_action_val)


def generate_behavior_flow_action(behavior_flow, states, config, deterministic=False):
    return behavior_flow.behavior_action(
        states=states,
        steps=config.flow_step,
        x_t_clip_value=config.x_t_clip_value,
        deterministic=deterministic,
    ).clamp(-config.max_action_val, config.max_action_val)


def energy_critic_update(batch, models, optimizers, config):
    energy_model = models["energy_model"]
    behavior_flow = models["behavior_flow"]
    energy_q_optimizer = optimizers["energy_q_optimizer"]
    energy_v_optimizer = optimizers.get("energy_v_optimizer")

    q_loss, v_loss, loss_info = energy_model.loss(
        tau=batch.get("iql_tau", getattr(config, "iql_tau")),
        behavior_model=behavior_flow,
        observations=batch["observations"],
        actions=batch["actions"],
        next_observations=batch["next_observations"],
        next_actions=batch.get("next_actions"),
        rewards=batch["rewards"],
        dones=batch["dones"],
        fake_actions=batch.get("fake_actions"),
        fake_next_actions=batch.get("fake_next_actions"),
    )
    if v_loss is not None and energy_v_optimizer is not None:
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(energy_model.v.parameters(), max_norm=10.0)
        energy_v_optimizer.step()
        energy_v_optimizer.zero_grad()
    if q_loss is not None:
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(energy_model.q.parameters(), max_norm=10.0)
        energy_q_optimizer.step()
        energy_q_optimizer.zero_grad()
    energy_model.q_ema()
    return loss_info


def behavior_flow_update(batch, models, optimizers, config):
    behavior_flow = models["behavior_flow"]
    energy_model = models["energy_model"]
    bf_optimizer = optimizers["bf_optimizer"]

    observations = batch["observations"]
    actions = batch["actions"]
    x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
        argus=config,
        observations=observations,
        actions=actions,
        energy=None,
        beta=config.beta,
        energy_model=energy_model,
        weighted_samples_type=WeightedSamplesType.linear_interpolation,
    )
    v_pred = behavior_flow(torch.cat([observations, x_t], dim=-1), t)
    loss = F.mse_loss(v_pred, dx_dt)
    bf_optimizer.zero_grad()
    loss.backward()
    bf_optimizer.step()
    return {"behavior_flow_loss": _item(loss)}


def adv_value_update(batch, models, optimizers, config):
    flow_energy_model = models["flow_energy_model"]
    energy_model = models["energy_model"]
    fv_optimizer = optimizers["fv_optimizer"]
    fv_v_optimizer = optimizers["fv_v_optimizer"]

    observations = batch["observations"]
    actions = batch["actions"]
    x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
        argus=config,
        observations=observations,
        actions=actions,
        energy=None,
        beta=config.beta,
        energy_model=energy_model,
        weighted_samples_type=WeightedSamplesType.linear_interpolation,
        clip_value=config.x_t_clip_value,
    )
    q_value = flow_energy_model.q(x=torch.cat([observations, x_t, dx_dt], dim=-1), t=t)
    normed_dx_dt = dx_dt / torch.norm(dx_dt, dim=-1, keepdim=True).clamp_min(1e-6)
    v_value = flow_energy_model.v(x=torch.cat([observations, x_t, normed_dx_dt], dim=-1), t=t)
    with torch.no_grad():
        target_fv = energy_model.get_scaled_q(obs=observations, act=actions, scale=config.energy_scale)
    q_loss = F.mse_loss(q_value, target_fv.detach())
    v_loss = F.mse_loss(v_value, target_fv.detach())

    fv_optimizer.zero_grad()
    q_loss.backward()
    fv_optimizer.step()
    fv_v_optimizer.zero_grad()
    v_loss.backward()
    fv_v_optimizer.step()
    _ema_update(flow_energy_model.q_target, flow_energy_model.q, config.ema_decay)
    return {
        "flow_value_loss": _item(q_loss),
        "flow_v_value_loss": _item(v_loss),
        "flow_value": _item(q_value.mean()),
        "flow_v_value": _item(v_value.mean()),
    }


def adv_policy_update(batch, models, optimizers, config):
    flow_model = models["flow_model"]
    target_flow_model = models["target_flow_model"]
    flow_energy_model = models["flow_energy_model"]
    energy_model = models["energy_model"]
    flow_optimizer = optimizers["flow_optimizer"]
    fv_optimizer = optimizers.get("fv_optimizer")

    observations = batch["observations"]
    actions = batch["actions"]
    x_t, t, dx_dt, _ = sample_weighted_interpolated_points(
        argus=config,
        observations=observations,
        actions=actions,
        energy=None,
        beta=config.beta,
        energy_model=energy_model,
        weighted_samples_type=WeightedSamplesType.linear_interpolation,
        clip_value=config.x_t_clip_value,
    )
    pred_u = flow_model(x=torch.cat([observations, x_t], dim=-1), t=t)
    pred_u = pred_u.clamp(-config.x_t_clip_value, config.x_t_clip_value)
    divergence = F.mse_loss(pred_u, dx_dt)
    normed_pred_u = pred_u / torch.norm(pred_u, dim=-1, keepdim=True).clamp_min(1e-6)
    pred_q = flow_energy_model.q(x=torch.cat([observations, x_t, pred_u], dim=-1), t=t)
    pred_v = flow_energy_model.v(x=torch.cat([observations, x_t, normed_pred_u], dim=-1), t=t)
    advantage = pred_q - pred_v.detach()
    if getattr(config, "adv_batch_norm", False):
        adv_mean = advantage.mean().detach()
        adv_std = advantage.std(unbiased=False).detach()
        advantage = (advantage - adv_mean) / (adv_std + 1e-6)
    loss = config.divergence_coef * divergence - advantage.mean()

    flow_optimizer.zero_grad()
    loss.backward()
    flow_optimizer.step()
    if fv_optimizer is not None:
        fv_optimizer.zero_grad()
    _ema_update(target_flow_model, flow_model, config.ema_decay)
    return {
        "flow_divergence_coef": config.divergence_coef,
        "flow_divergence": _item(divergence.mean()),
        "flow_loss": _item(loss),
        "flow_adv_mean": _item((pred_q - pred_v.detach()).mean()),
        "flow_adv_std": _item((pred_q - pred_v.detach()).std(unbiased=False)),
    }
