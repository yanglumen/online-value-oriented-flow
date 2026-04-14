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
from trainer.adaptive_flow_step_trainer import guided_flow_trainer

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

def import_parameters(var_kwargs=None, mode="adaptive_flow_step_parameters"):
    print(colored(f"Loading {mode} parameter ......", color="green"))
    if mode == "adaptive_flow_step_parameters":
        hyperparameters = adaptive_flow_step_parameters
        hyperparameters["mode"] = "guided_flow"
    else:
        raise Exception("The mode of import_parameters is wrong !!!")
    for key, val in base_parameters.items():
        if key not in hyperparameters.keys():
            hyperparameters[key] = val
    for enumerate_key, enumerate_var in {
        "flow_guided_mode": FlowGuidedMode, "critic_type": CriticType,
        "expectile_type": ExpectileMode, "rl_mode": RLTrainMode,
    }.items():
        if enumerate_key in var_kwargs.keys():
            var_kwargs[enumerate_key] = enumerate_var[var_kwargs[enumerate_key]]
    if var_kwargs:
        hyperparameters.update(var_kwargs)
    return hyperparameters

def hyperparameter_finetuning(argus):
    if argus.rl_mode in [RLTrainMode.flow_constrained_rl, RLTrainMode.flow_constrained_rl2, RLTrainMode.flow_constrained_rl3,
                         RLTrainMode.flow_constrained_rl4, RLTrainMode.flow_constrained_rl5]:
        if argus.dataset == "halfcheetah-medium-expert-v2":
            argus.update_energy_end_epoch = argus.epoch_offset
            argus.update_behavior_end_epoch = argus.epoch_offset
            argus.update_flow_start_epoch = argus.epoch_offset
    return argus

def train(**var_kwargs):
    argus = dict2obj(import_parameters(var_kwargs, mode="adaptive_flow_step_parameters"))
    argus = hyperparameter_finetuning(argus=argus)
    argus = generate_wandb_exp_name(argus=argus, wandb_project=argus.mode)
    argus = seed_configuration(argus=argus)
    print(obj2dict(argus))
    dataset = InDistributionSequenceDataset(
        argus=argus, project_path=None, env_name=argus.dataset, domain=argus.domain, sequence_length=argus.sequence_length,
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

    trainer = guided_flow_trainer(
        argus=argus, train_flow=train_flow, behavior_flow=behavior_flow, dataset=dataset)

    trainer.guided_train(num_epochs=200, num_steps_per_epoch=10000)

if __name__ == "__main__":
    fire.Fire(train)
