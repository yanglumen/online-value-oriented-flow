import torch
import enum

class WeightedSamplesType(enum.Enum):
    subopt2opt = enum.auto()
    subopt_state2opt_state = enum.auto()
    linear_interpolation = enum.auto()
    noise_interpolation = enum.auto()
    noise_action = enum.auto()
    specific_obs_x0x1 = enum.auto()
    obs_linear_interpolation = enum.auto()

    def __str__(self):
        return self.name

class CriticType(enum.Enum):
    iql = enum.auto()
    ciql = enum.auto()
    cql = enum.auto()

    def __str__(self):
        return self.name

class FlowGuidedMode(enum.Enum):
    normal = enum.auto()
    gradient = enum.auto()
    expectile = enum.auto()

    def __str__(self):
        return self.name

class RLTrainMode(enum.Enum):
    none = enum.auto()
    use_rl_q = enum.auto()
    adv_rl = enum.auto()
    full_rl_like = enum.auto()
    flow_constrained_rl = enum.auto()
    flow_constrained_rl2 = enum.auto()
    flow_constrained_rl3 = enum.auto()
    flow_constrained_rl4 = enum.auto()
    flow_constrained_rl5 = enum.auto()
    dist_flow_constrained_rl = enum.auto()
    direct_flow_2_result = enum.auto()
    use_rl_q_dist_time = enum.auto()
    grpo = enum.auto()
    adaptive_step = enum.auto()

base_parameters = dict(
    mode = None,
    wandb_log = False,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = None,
    reset_seed = False,
    seed = 661,
    device = "cuda",
)


base_flow_parameters = dict(
    # todo    dataset setting
    dataset='2spirals',
    # todo       ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]
    current_exp_label = "toy_test",
    datanum = 2000000,
    lr = 0.0003,
    batch_size = 1024,
    beta = 5.0,
    time_rescale = False,
    multi_stage_genration = 2,

    x_record=True,
    flow_step_scale = 1.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_interpolation, WeightedSamplesType.subopt2opt]
)


f2b_parameters = dict(
    debug_mode=True,
    # todo    dataset setting
    domain="mujoco",
    dataset='hopper-medium-v2',
    sequence_length=1,
    normalizer='GaussianNormalizer',
    termination_penalty = 0,
    discount=0.99,
    flow_step=10,
    x_t_clip_value = 10.0,
    multi_mode_action_evaluation=False,
    returns_scale=1,
    reward_tune = 'iql_antmaze',
    CEP_dataset_load_mode = False,
    multi_etas = [],
    eta = 0,
    train_with_normed_data=True,
    rl_mode = RLTrainMode.adaptive_step,
    # todo evaluation
    eval_episodes = 10,
    # todo training
    flow_guided_mode = FlowGuidedMode.normal,
    critic_type = CriticType.ciql,
    large_q0_model=False,
    iql_tau = 0.5,
    ema_decay = 0.995,
    save_path = './output',
    load_critic_model = 'none',
    save_freq = 50000,
    energy_scale = 200,
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
    batch_size = 512,
    beta = 5.0,
    time_rescale = False,
    multi_stage_genration = 2,

    x_record=True,
    flow_step_scale = 1.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_action, WeightedSamplesType.subopt2opt]
)


