import torch
import enum

class WeightedSamplesType(enum.Enum):
    subopt2opt = enum.auto()
    linear_interpolation = enum.auto()
    noise_interpolation = enum.auto()
    random_direction = enum.auto()

    def __str__(self):
        return self.name

class FlowGuidedMode(enum.Enum):
    normal = enum.auto()
    gradient = enum.auto()
    expectile = enum.auto()
    expectile_rl = enum.auto()

    def __str__(self):
        return self.name

base_parameters = dict(
    mode = None,
    wandb_log = False,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = None,
    reset_seed = True,
    seed = 0,
    device = "cuda",
)


base_flow_parameters = dict(
    # todo    dataset setting
    dataset='swissroll',
    # todo       ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]
    current_exp_label = "toy_test",
    datanum = 2000000,
    lr = 0.0003,
    batch_size = 1024,
    beta = 5.0,
    time_rescale = True,
    multi_stage_genration = 2,
    flow_steps = 50,
    tau = 0.9,
    flow_guided_mode = FlowGuidedMode.expectile_rl,

    x_record=True,
    flow_step_scale = 1.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_interpolation, WeightedSamplesType.subopt2opt]
)


guided_flow_parameters = dict(
    debug_mode=True,
    # todo    dataset setting
    domain="mujoco",
    dataset='hopper-medium-v2',
    sequence_length=1,
    normalizer='GaussianNormalizer',
    termination_penalty = 0,
    discount=0.99,
    returns_scale=400,
    reward_tune = 'iql_antmaze',
    CEP_dataset_load_mode = False,
    multi_etas = [],
    eta = 0,
    train_with_normed_data=False,
    # todo evaluation
    eval_episodes = 10,
    # todo training
    large_q0_model=False,
    iql_tau = 0.5,
    ema_decay = 0.995,
    save_path = './output',
    save_freq = 50000,
    energy_scale = 400,
    wandb_log = False,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = None,
    # todo       halfcheetah    hopper     walker2d               random   medium  expert   medium-expert   medium-replay  full-replay                  v2
    # todo       antmaze        umaze  umaze-diverse medium-play medium-diverse large-play large-diverse         v1
    # todo       maze2d         umaze medium large umaze-dense medium-dense large-dense                          v0 v1
    current_exp_label = "test",
    datanum = 2000000,
    lr = 0.0003,
    batch_size = 1024,
    beta = 5.0,
    time_rescale = True,
    multi_stage_genration = 2,

    x_record=True,
    flow_step_scale = 1.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_interpolation, WeightedSamplesType.subopt2opt]
)




