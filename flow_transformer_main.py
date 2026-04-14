import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy as np
import fire
from config.dict2class import *
from config.flow_transformer_hyperparameter import *
from termcolor import colored
from trainer.trainer_util import seed_configuration
from datasets_process.sequence_dataset import FlowTFSequenceDataset
from models.flow_model import FlowMatchingNet, LargeFlowMatchingNet
from models.flow_transformer import FlowTransformer, QFlowTransformer
from trainer.flow_transformer_trainer import flow_tf_trainer
from models.energy_model import iql_critic

from models.energy_model import iql_critic, ciql_critic, in_support_softmax_q_learning_critic
from models.flow_value_model import iql_flow_critic, ciql_flow_critic, adv_decision_flow_iql_flow_critic
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

def import_parameters(var_kwargs=None, mode="flow_transformer_hyperparameter"):
    print(colored(f"Loading {mode} parameter ......", color="green"))
    if mode == "flow_transformer_hyperparameter":
        hyperparameters = flow_transformer_hyperparameters
        hyperparameters["mode"] = "flow_transformer"
    else:
        raise Exception("The mode of import_parameters is wrong !!!")
    for key, val in base_parameters.items():
        if key not in hyperparameters.keys():
            hyperparameters[key] = val
    for enumerate_key, enumerate_var in {
        "flow_guided_mode": FlowGuidedMode, "critic_type": CriticType, "expectile_type": ExpectileMode, "rl_mode": RLTrainMode,
    }.items():
        if enumerate_key in var_kwargs.keys():
            var_kwargs[enumerate_key] = enumerate_var[var_kwargs[enumerate_key]]
    if var_kwargs:
        hyperparameters.update(var_kwargs)
    return hyperparameters

def hyperparameter_finetuning(argus):
    if argus.rl_mode in [RLTrainMode.vanilla_flow_tf]:
        pass
    return argus

def train(**var_kwargs):
    argus = dict2obj(import_parameters(var_kwargs, mode="flow_transformer_hyperparameter"))
    argus = hyperparameter_finetuning(argus=argus)
    argus = generate_wandb_exp_name(argus=argus, wandb_project=argus.mode)
    argus = seed_configuration(argus=argus)
    print(obj2dict(argus))
    dataset = FlowTFSequenceDataset(
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
    behavior_flow = FlowMatchingNet(
        argus=argus, input_dim=argus.observation_dim+argus.action_dim, output_dim=argus.action_dim)

    if argus.rl_mode == RLTrainMode.vanilla_flow_tf:
        train_flow = FlowTransformer(argus=argus, state_dim=argus.observation_dim, action_dim=argus.action_dim)
        energy_model = None
    else:
        train_flow = QFlowTransformer(argus=argus, state_dim=argus.observation_dim, action_dim=argus.action_dim)
        energy_model = iql_critic(adim=argus.action_dim, sdim=argus.observation_dim, args=argus)

    trainer = flow_tf_trainer(
        argus=argus, train_flow=train_flow, behavior_flow=behavior_flow, energy_model=energy_model, dataset=dataset)

    if argus.rl_mode == RLTrainMode.vanilla_flow_tf:
        trainer.train(num_epochs=300, num_steps_per_epoch=10000)
    elif argus.rl_mode == RLTrainMode.Q_value_flow_tf:
        trainer.Q_value_guided_train(num_epochs=300, num_steps_per_epoch=10000)
    else:
        raise Exception("The rl_mode is wrong !!!")
    # trainer.energy_gradient_guided_train(num_epochs=100, num_steps_per_epoch=20000)

if __name__ == "__main__":
    fire.Fire(train)
