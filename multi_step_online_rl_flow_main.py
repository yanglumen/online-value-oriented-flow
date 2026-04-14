import fire
from termcolor import colored

from config.dict2class import dict2obj, obj2dict
from config.multistep_rl_flow_hyperparameter import (
    base_parameters,
    multistep_rl_flow_parameters,
    CriticType,
    FlowGuidedMode,
    ExpectileMode,
    RLTrainMode,
)
from datasets_process.online_sequence_dataset import OnlineSequenceDataset
from models.flow_model import FlowMatchingNet, LargeFlowMatchingNet
from models.flow_value_model import iql_flow_critic
from models.sliding_window_iql_critic import sliding_window_iql_critic
from trainer.multi_step_online_rl_flow_trainer import online_guided_flow_trainer
from trainer.trainer_util import seed_configuration


online_multistep_rl_flow_parameters = {
    **multistep_rl_flow_parameters,
    "mode": "online_guided_flow",
    "domain": "gymnasium",
    "dataset": "HalfCheetah-v5",
    "critic_type": CriticType.iql,
    "rl_mode": RLTrainMode.adv_rl,
    "sequence_length": 2,
    "online_use_sliding_window_critic": True,
    "swdg_num_q_ensembles": 8,
    "swdg_window_size": 4,
    "swdg_window_step": 1,
    "swdg_use_diversity_reg": False,
    "swdg_diversity_coef": 0.0,
    "lr": 0.00005,
    "batch_size": 256,
    "divergence_coef": 3.0,
    "online_adv_batch_norm": True,
    "multi_mode_action_evaluation": False,
    "online_action_noise_std": 0.3,
    "online_action_noise_clip": 1.0,
    "online_action_noise_enable": True,
    "online_eval_deterministic": True,
    "normalizer": "OnlineGaussianNormalizer",
    "debug_mode": False,
    "wandb_log": True,
    "wandb_project_name": "online_guided_flow",
    "wandb_log_frequency": 100,
    "online_print_frequency": 200,
    "online_behavior_only": False,
    "reward_scale": 1.0,
    "preserve_ep": 300,
    "update_flow_start_epoch": 10,
    "online_behavior_bootstrap_updates": -1,
    "online_min_flow_updates_before_deploy": 1,
    "online_deploy_flow_after_updates": 1000,
    "online_deploy_flow_after_epoch": 20,
    "online_gradual_deploy_enable": False,
    "online_gradual_deploy_start_prob": 0.1,
    "online_gradual_deploy_end_prob": 1.0,
    "online_gradual_deploy_ramp_updates": 5000,
    "online_init_train_flow_from_behavior": True,
    "online_epochs": 100,
    "online_rollout_steps_per_epoch": 1000,
    "online_updates_per_epoch": 1000,
    "online_init_steps": 5000,
    "online_random_steps": 5000,
    "online_eval_freq": 5000,
    "load_offline_checkpoint": False,
    "online_eval_policy_stochastic": False,
    "online_late_phase_start_epoch": -1,
    "online_late_lr_scale": 1.0,
    "online_late_divergence_scale": 1.0,
    "online_late_update_ratio_scale": 1.0,
}


def generate_wandb_exp_name(argus, wandb_project):
    argus.wandb_exp_name = f"{argus.dataset}-{argus.current_exp_label}"
    argus.wandb_exp_group = argus.dataset
    if argus.wandb_project_name is None:
        argus.wandb_project_name = wandb_project
    return argus

def import_parameters(var_kwargs=None, mode="online_multistep_rl_flow_parameters"):
    print(colored(f"Loading {mode} parameter ......", color="green"))
    if mode != "online_multistep_rl_flow_parameters":
        raise Exception("The mode of import_parameters is wrong !!!")
    hyperparameters = dict(online_multistep_rl_flow_parameters)
    for key, val in base_parameters.items():
        if key not in hyperparameters:
            hyperparameters[key] = val
    enum_mapping = {
        "flow_guided_mode": FlowGuidedMode,
        "critic_type": CriticType,
        "expectile_type": ExpectileMode,
        "rl_mode": RLTrainMode,
    }
    if var_kwargs is None:
        var_kwargs = {}
    for enumerate_key, enumerate_var in enum_mapping.items():
        if enumerate_key in var_kwargs:
            var_kwargs[enumerate_key] = enumerate_var[var_kwargs[enumerate_key]]
    hyperparameters.update(var_kwargs)
    return hyperparameters


def hyperparameter_finetuning(argus):
    if argus.domain not in ["gym", "gymnasium"]:
        raise ValueError(
            "Online entrypoint currently supports modern Mujoco gymnasium environments only. "
            "Use domain='gymnasium'."
        )
    if argus.domain == "gym":
        argus.domain = "gymnasium"
    if argus.critic_type != CriticType.iql:
        raise ValueError(
            "The online entrypoint is now standardized to critic_type='iql'."
        )
    if argus.online_behavior_only:
        argus.rl_mode = RLTrainMode.adv_rl
    elif argus.rl_mode != RLTrainMode.adv_rl:
        raise ValueError(
            "The online entrypoint now supports rl_mode='adv_rl' only. "
            "Use --online_behavior_only True for the behavior-flow baseline."
        )
    return argus


def build_energy_model(argus):
    if argus.critic_type != CriticType.iql:
        raise ValueError("Online training only supports the IQL critic.")
    if getattr(argus, "online_use_sliding_window_critic", True):
        return sliding_window_iql_critic(adim=argus.action_dim, sdim=argus.obs_embed_dim, args=argus)
    raise ValueError("Online training is standardized to use the sliding-window IQL critic.")


def build_flow_energy_model(argus):
    if argus.rl_mode != RLTrainMode.adv_rl:
        raise ValueError("Online training only supports the adv_rl flow critic.")
    return iql_flow_critic(
        adim=argus.action_dim, sdim=argus.observation_dim + argus.action_dim,
        args=argus, use_TwinQ=argus.flow_value_TwinQ)


def train(**var_kwargs):
    argus = dict2obj(import_parameters(var_kwargs, mode="online_multistep_rl_flow_parameters"))
    argus = hyperparameter_finetuning(argus=argus)
    argus = generate_wandb_exp_name(argus=argus, wandb_project=argus.mode)
    argus = seed_configuration(argus=argus)
    print(obj2dict(argus))

    dataset = OnlineSequenceDataset(
        argus=argus, project_path=None, env_name=argus.dataset, domain=argus.domain,
        sequence_length=argus.sequence_length, normalizer=argus.normalizer,
        termination_penalty=argus.termination_penalty, discount=argus.discount,
        returns_scale=argus.returns_scale)
    argus.observation_dim = dataset.observation_dim
    argus.action_dim = dataset.action_dim
    argus.obs_embed_dim = dataset.observation_dim
    argus.act_embed_dim = dataset.action_dim
    argus.input_channels = dataset.observation_dim
    argus.out_channels = dataset.observation_dim
    argus.max_action_val = dataset.original_action_range[1]

    FlowModel = LargeFlowMatchingNet if argus.large_flow else FlowMatchingNet
    behavior_flow = FlowMatchingNet(
        argus=argus, input_dim=argus.observation_dim + argus.action_dim, output_dim=argus.action_dim
    )
    train_flow = FlowModel(
        argus=argus, input_dim=argus.observation_dim + argus.action_dim, output_dim=argus.action_dim)
    target_train_flow = FlowModel(
        argus=argus, input_dim=argus.observation_dim + argus.action_dim, output_dim=argus.action_dim)
    energy_model = build_energy_model(argus)
    flow_energy_model = build_flow_energy_model(argus)

    trainer = online_guided_flow_trainer(
        argus=argus, train_flow=train_flow, target_train_flow=target_train_flow,
        behavior_flow=behavior_flow, flow_energy_model=flow_energy_model,
        energy_model=energy_model, dataset=dataset)

    if argus.load_offline_checkpoint:
        trainer.load_critic_checkpoint(
            loadpath=f"{argus.save_path}/guided_flow/{argus.domain}/{'_'.join(argus.dataset.split('-'))}/{argus.load_critic_model}")

    trainer.online_train(
        num_epochs=argus.online_epochs,
        rollout_steps_per_epoch=argus.online_rollout_steps_per_epoch,
        num_updates_per_epoch=argus.online_updates_per_epoch)


if __name__ == "__main__":
    fire.Fire({"train": train})
