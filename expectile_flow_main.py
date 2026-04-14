import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import numpy as np
import fire
from config.dict2class import *
from config.expectile_flow_hyperparameter import *
from termcolor import colored
from trainer.trainer_util import seed_configuration
from datasets_process.sequence_dataset import InDistributionSequenceDataset
from models.flow_model import FlowMatchingNet
from models.energy_model import iql_critic, ciql_critic
from models.expectile_model import expectile_Q, expectile_V, expectile_time_func
from trainer.expectile_flow_model_trainer import guided_flow_trainer


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

def import_parameters(var_kwargs=None, mode="expectile_guided_flow_parameters"):
    print(colored(f"Loading {mode} parameter ......", color="green"))
    if mode == "expectile_guided_flow_parameters":
        hyperparameters = expectile_guided_flow_parameters
        hyperparameters["mode"] = "guided_flow"
    else:
        raise Exception("The mode of import_parameters is wrong !!!")
    for key, val in base_parameters.items():
        if key not in hyperparameters.keys():
            hyperparameters[key] = val
    for enumerate_key, enumerate_var in {
        "flow_guided_mode": FlowGuidedMode, "critic_type": CriticType, "expectile_type": ExpectileMode,
    }.items():
        if enumerate_key in var_kwargs.keys():
            var_kwargs[enumerate_key] = enumerate_var[var_kwargs[enumerate_key]]
    if var_kwargs:
        hyperparameters.update(var_kwargs)
    return hyperparameters

def train(**var_kwargs):
    argus = dict2obj(import_parameters(var_kwargs, mode="expectile_guided_flow_parameters"))
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
    flow_model = FlowMatchingNet(argus=argus, input_dim=argus.observation_dim+argus.action_dim, output_dim=argus.action_dim)
    if argus.critic_type == CriticType.iql:
        energy_model = iql_critic(adim=argus.action_dim, sdim=argus.obs_embed_dim, args=argus)
    elif argus.critic_type == CriticType.ciql:
        energy_model = ciql_critic(adim=argus.action_dim, sdim=argus.obs_embed_dim, args=argus)
    else:
        raise Exception("The critic_type is wrong !!!")
    expectile_q = expectile_time_func(input_dim=argus.observation_dim+argus.action_dim+argus.action_dim)
    expectile_v = expectile_time_func(input_dim=argus.observation_dim+argus.action_dim)
    trainer = guided_flow_trainer(
        argus=argus, model=flow_model, energy_model=energy_model, expectile_q=expectile_q,
        expectile_v=expectile_v, dataset=dataset)
    trainer.guided_train(num_epochs=100, num_steps_per_epoch=10000)
    # trainer.energy_gradient_guided_train(num_epochs=100, num_steps_per_epoch=20000)

if __name__ == "__main__":
    fire.Fire(train)