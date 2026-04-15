import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import fire
from config.dict2class import *
from config.multistep_rl_flow_hyperparameter import *
from termcolor import colored
from trainer.trainer_util import seed_configuration
from datasets_process.sequence_dataset import InDistributionSequenceDataset
from models.flow_model import FlowMatchingNet, LargeFlowMatchingNet
from models.energy_model import iql_critic, ciql_critic, in_support_softmax_q_learning_critic
from models.flow_value_model import iql_flow_critic, ciql_flow_critic, adv_decision_flow_iql_flow_critic
from trainer.multistep_rl_flow_model_trainer import guided_flow_trainer
from models.flow_constrained_energy_model import flow_constrained_iql_critic, direct_flow2result_iql_critic
# from models.multistep_rl_flow_model import rl_flow_value_func

def generate_wandb_exp_name(argus, wandb_project):  # diffusion_q_function, denoise_RL
	argus.wandb_exp_name = argus.dataset
	argus.wandb_exp_name = f'{argus.wandb_exp_name}-{argus.current_exp_label}' # {random.randint(int(1e5), int(1e6) - 1)}
	if len(argus.dataset.split("-")) > 2:
		group_name = "-".join(argus.dataset.split("-")[0:2])
	else:
		group_name = argus.dataset
	argus.wandb_exp_group = group_name
	argus.wandb_project_name = wandb_project
	return argus

def import_parameters(var_kwargs=None, mode="multistep_rl_flow_parameters"):
    print(colored(f"Loading {mode} parameter ......", color="green"))
    if mode == "multistep_rl_flow_parameters":
        hyperparameters = dict(multistep_rl_flow_parameters)
        hyperparameters["mode"] = "guided_flow"
        hyperparameters.pop("adv_rl_multiple_actions", None)
    else:
        raise Exception("The mode of import_parameters is wrong !!!")
    if var_kwargs is None:
        var_kwargs = {}
    removed_args = {
        "adv_rl_multiple_actions": "adv_rl/grpo value updates now use the shared single-sample flow core; this argument is no longer read.",
    }
    stale_args = sorted(set(var_kwargs) & set(removed_args))
    if stale_args:
        details = "; ".join(f"{key}: {removed_args[key]}" for key in stale_args)
        raise ValueError(f"Removed or unused offline argument(s): {details}")
    for key, val in base_parameters.items():
        if key not in hyperparameters.keys():
            hyperparameters[key] = val
    for enumerate_key, enumerate_var in {
        "flow_guided_mode": FlowGuidedMode, "critic_type": CriticType,
        "expectile_type": ExpectileMode, "rl_mode": RLTrainMode,
    }.items():
        if enumerate_key in var_kwargs.keys():
            var_kwargs[enumerate_key] = enumerate_var[var_kwargs[enumerate_key]]
    unknown_args = sorted(set(var_kwargs) - set(hyperparameters))
    if unknown_args:
        raise ValueError(f"Unknown offline argument(s): {unknown_args}")
    if var_kwargs:
        hyperparameters.update(var_kwargs)
    return hyperparameters

def hyperparameter_finetuning(argus):
    argus.sequence_length = int(argus.sequence_length)
    if argus.sequence_length < 1:
        raise ValueError("Offline sequence_length must be >= 1.")
    if argus.rl_mode in [RLTrainMode.flow_constrained_rl, RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl3,
                         RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
        if argus.dataset == "halfcheetah-medium-expert-v2":
            argus.update_energy_end_epoch = argus.epoch_offset
            argus.update_behavior_end_epoch = argus.epoch_offset
            argus.update_flow_start_epoch = argus.epoch_offset
    return argus

def train(**var_kwargs):
    argus = dict2obj(import_parameters(var_kwargs, mode="multistep_rl_flow_parameters"))
    argus = hyperparameter_finetuning(argus=argus)
    argus = generate_wandb_exp_name(argus=argus, wandb_project=argus.mode)
    argus = seed_configuration(argus=argus)
    print(obj2dict(argus))
    dataset = InDistributionSequenceDataset(
        argus=argus, project_path=None, env_name=argus.dataset, domain=argus.domain,
        sequence_length=argus.sequence_length,
        normalizer=argus.normalizer, termination_penalty=argus.termination_penalty, discount=argus.discount,
        returns_scale=argus.returns_scale)
    argus.observation_dim = dataset.observation_dim
    argus.action_dim = dataset.action_dim
    argus.obs_embed_dim = dataset.observation_dim
    argus.act_embed_dim = dataset.action_dim
    argus.input_channels = dataset.observation_dim
    argus.out_channels = dataset.observation_dim
    argus.max_action_val = dataset.max_action_val
    if argus.large_flow:
        FlowModel = LargeFlowMatchingNet
    else:
        FlowModel = FlowMatchingNet
    if argus.rl_mode in [RLTrainMode.flow_constrained_rl5]:
        behavior_flow = LargeFlowMatchingNet(
            argus=argus, input_dim=argus.observation_dim + argus.action_dim, output_dim=argus.action_dim)
    else:
        behavior_flow = FlowMatchingNet(
            argus=argus, input_dim=argus.observation_dim+argus.action_dim, output_dim=argus.action_dim)
    train_flow = FlowModel(
        argus=argus, input_dim=argus.observation_dim + argus.action_dim, output_dim=argus.action_dim)
    target_train_flow = FlowModel(
        argus=argus, input_dim=argus.observation_dim + argus.action_dim, output_dim=argus.action_dim)
    if argus.critic_type == CriticType.iql:
        energy_model = iql_critic(adim=argus.action_dim, sdim=argus.obs_embed_dim, args=argus)
    elif argus.critic_type == CriticType.ciql:
        energy_model = ciql_critic(adim=argus.action_dim, sdim=argus.obs_embed_dim, args=argus)
    elif argus.critic_type == CriticType.isql:
        energy_model = in_support_softmax_q_learning_critic(adim=argus.action_dim, sdim=argus.obs_embed_dim, args=argus)
    else:
        raise Exception("The critic_type is wrong !!!")

    if argus.rl_mode in [RLTrainMode.flow_constrained_rl, RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl3,
                         RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
        flow_energy_model = flow_constrained_iql_critic(adim=argus.action_dim, sdim=argus.observation_dim, args=argus)
    elif argus.rl_mode in [RLTrainMode.direct_flow_2_result]:
        flow_energy_model = direct_flow2result_iql_critic(adim=argus.action_dim, sdim=argus.observation_dim, args=argus)
    elif argus.rl_mode in [RLTrainMode.adv_rl, RLTrainMode.grpo]:
        flow_energy_model = iql_flow_critic(adim=argus.action_dim, sdim=argus.observation_dim + argus.action_dim, args=argus, use_TwinQ=argus.flow_value_TwinQ)
    else:
        flow_energy_model = iql_flow_critic(adim=argus.action_dim, sdim=argus.observation_dim + argus.action_dim, args=argus, use_TwinQ=argus.flow_value_TwinQ)
    # flow_value_func = rl_flow_value_func(input_dim=argus.observation_dim + argus.action_dim)
    # target_flow_value_func = rl_flow_value_func(input_dim=argus.observation_dim + argus.action_dim)

    # flow_value_func = rl_flow_value_func(input_dim=argus.observation_dim+argus.action_dim+argus.action_dim)
    # target_flow_value_func = rl_flow_value_func(input_dim=argus.observation_dim+argus.action_dim+argus.action_dim)
    # flow_v_value = rl_flow_value_func(input_dim=argus.observation_dim+argus.action_dim+argus.action_dim)
    trainer = guided_flow_trainer(
        argus=argus, train_flow=train_flow, target_train_flow=target_train_flow, behavior_flow=behavior_flow,
        flow_energy_model=flow_energy_model,
        # flow_value_func=flow_value_func, target_flow_value_func=target_flow_value_func, flow_v_value=flow_v_value,
        energy_model=energy_model, dataset=dataset)

    if argus.rl_mode == RLTrainMode.use_rl_q:
        trainer.guided_train(num_epochs=200, num_steps_per_epoch=10000)
    elif argus.rl_mode == RLTrainMode.adv_rl:
        trainer.adv_based_flow_train(num_epochs=300, num_steps_per_epoch=10000)
        # trainer.visualization_intermediate_actions()
        # trainer.adv_based_flow_train(num_epochs=300, num_steps_per_epoch=10000)
    elif argus.rl_mode == RLTrainMode.grpo:
        trainer.grpo_flow_train(num_epochs=300, num_steps_per_epoch=10000)
    elif argus.rl_mode == RLTrainMode.direct_flow_2_result:
        trainer.direct_flow2result_train(num_epochs=150, num_steps_per_epoch=10000)
    elif argus.rl_mode == RLTrainMode.full_rl_like:
        trainer.full_rl_like_train(num_epochs=150, num_steps_per_epoch=10000)
    elif argus.rl_mode in [RLTrainMode.flow_constrained_rl, RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl3,
                           RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
        trainer.flow_constrained_train(num_epochs=300, num_steps_per_epoch=10000)
    else:
        raise Exception("The rl_mode is wrong !!!")
    # trainer.energy_gradient_guided_train(num_epochs=100, num_steps_per_epoch=20000)

if __name__ == "__main__":
    fire.Fire(train)
