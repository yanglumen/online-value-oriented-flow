import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import numpy as np
import fire
from toy_example.config.dict2class import *
from toy_example.config.hyperparameter import *
from termcolor import colored
from trainer.trainer_util import seed_configuration
from toy_example.dataset_process.toy_dataset import ToyDataset
from toy_example.flow_model.toy_flow_model import FlowMatchingNet
from toy_example.flow_model.toy_energy_model import EnergyNet
from toy_example.toy_trainer.flow_model_trainer import toy_flow_trainer

def generate_wandb_exp_name(argus, wandb_project):  # diffusion_q_function, denoise_RL
	argus.wandb_exp_name = "toy_example"
	argus.wandb_exp_name = f'{argus.wandb_exp_name}-{argus.current_exp_label}' # {random.randint(int(1e5), int(1e6) - 1)}
	if len(argus.dataset.split("-")) > 2:
		group_name = "-".join(argus.dataset.split("-")[0:2])
	else:
		group_name = argus.dataset
	argus.wandb_exp_group = group_name
	argus.wandb_project_name = wandb_project
	return argus

def import_parameters(var_kwargs=None, mode="base_flow_parameters"):
    print(colored(f"Loading {mode} parameter ......", color="green"))
    if mode == "base_flow_parameters":
        hyperparameters = base_flow_parameters
        hyperparameters["mode"] = "base_flow_test"
    else:
        raise Exception("The mode of import_parameters is wrong !!!")
    for key, val in base_parameters.items():
        if key not in hyperparameters.keys():
            hyperparameters[key] = val
    if var_kwargs:
        hyperparameters.update(var_kwargs)
    return hyperparameters

def train(**var_kwargs):
    argus = dict2obj(import_parameters(var_kwargs, mode="base_flow_parameters"))
    argus = generate_wandb_exp_name(argus=argus, wandb_project=argus.mode)
    argus = seed_configuration(argus=argus)
    print(obj2dict(argus))
    dataset = ToyDataset(
        argus=argus, project_path=None, env_name=argus.dataset)
    argus.observation_dim = dataset.observation_dim
    argus.action_dim = dataset.action_dim
    flow_model = FlowMatchingNet(input_dim=argus.observation_dim)
    if argus.flow_guided_mode == FlowGuidedMode.normal:
        energy_model = EnergyNet(input_dim=argus.observation_dim)
    elif argus.flow_guided_mode == FlowGuidedMode.expectile_rl:
        energy_model = EnergyNet(input_dim=argus.observation_dim)
        expectile_energy_model = EnergyNet(input_dim=argus.observation_dim)
        direction_energy_model = EnergyNet(input_dim=argus.observation_dim+argus.observation_dim)
        energy_model = [energy_model, expectile_energy_model, direction_energy_model]
    else:
        raise Exception("The mode of import_parameters is wrong !!!")
    trainer = toy_flow_trainer(
        argus=argus, model=flow_model, energy_model=energy_model, dataset=dataset)
    trainer.guided_train(num_epochs=100, num_steps_per_epoch=10000)
    # trainer.energy_gradient_guided_train(num_epochs=100, num_steps_per_epoch=20000)

if __name__ == "__main__":
    fire.Fire(train)