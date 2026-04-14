import torch
import enum

class WeightedSamplesType(enum.Enum):
    subopt2opt = enum.auto()
    linear_interpolation = enum.auto()
    noise_interpolation = enum.auto()
    noise_action = enum.auto()
    specific_start_end = enum.auto()
    discrete_flow_step = enum.auto()
    same_t_step = enum.auto()
    multiple_linear_interpolation = enum.auto()
    adaptive_step_interpolation = enum.auto()

    def __str__(self):
        return self.name

class CriticType(enum.Enum):
    iql = enum.auto()
    ciql = enum.auto()
    cql = enum.auto()
    isql = enum.auto()

    def __str__(self):
        return self.name

class FlowPolicyType(enum.Enum):
    deterministic = enum.auto()
    ppo = enum.auto()

    def __str__(self):
        return self.name

class FlowGuidedMode(enum.Enum):
    normal = enum.auto()
    gradient = enum.auto()
    expectile = enum.auto()

    def __str__(self):
        return self.name

class ExpectileMode(enum.Enum):
    expectile = enum.auto()
    expectile_rl = enum.auto()

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
    dataset='2spirals',
    # todo       ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]
    current_exp_label = "toy_test",
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


#todo #############################  guided_flow ############################################
multistep_rl_flow_parameters = dict(
    # todo       halfcheetah    hopper     walker2d               random   medium  expert   medium-expert   medium-replay  full-replay                  v2
    # todo       antmaze        umaze  umaze-diverse medium-play medium-diverse large-play large-diverse         v1
    # todo       maze2d         umaze medium large umaze-dense medium-dense large-dense                          v0 v1
    locomotion_dataset = ["halfcheetah-full-replay-v2", "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
                          "hopper-full-replay-v2", "hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2",
                          "walker2d-full-replay-v2", "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2"],
    adroit_dataset = ["hammer-expert-v1", "hammer-human-v1", "hammer-cloned-v1", "pen-expert-v1", "pen-human-v1", "pen-cloned-v1",
                      "relocate-expert-v1", "relocate-human-v1", "relocate-cloned-v1", "door-expert-v1", "door-human-v1", "door-cloned-v1"],
    maze2d_dataset = ["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1", "maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"],
    antmaze_dataset = ["antmaze-umaze-v1", "antmaze-umaze-diverse-v1", "antmaze-medium-play-v1", "antmaze-medium-diverse-v1", "antmaze-large-play-v1", "antmaze-large-diverse-v1"],
    debug_mode=True,
    # todo    dataset setting
    domain="mujoco",
    dataset='halfcheetah-medium-expert-v2',
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
    flow_step = 10,
    direct_flow_step = 1,
    multi_mode_action_evaluation=False,
    # todo training
    flow_guided_mode = FlowGuidedMode.normal,
    critic_type = CriticType.iql,
    rl_mode = RLTrainMode.grpo,
    flow_value_TwinQ = False,
    large_q0_model=False,
    large_flow=False,
    large_flow_V=False,
    iql_tau = 0.5,
    expectile_func_tau = 0.9,
    expectile_flow_tau = 0.95,
    noise_scale_threshold = 0.1,
    noise_similarity_threshold = 0.85,
    divergence_coef = 1.0,
    conservative_coef = 1.0,
    divergence_discount = 1.0,
    weight_policy_regression_coef = 0.1,
    x_t_clip_value = 30.0,
    ema_decay = 0.995,
    save_path = './output',
    load_critic_model = 'ciql_rl_flow',
    save_freq = 50000,
    energy_scale = 1,
    wandb_log = False,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = None,
    current_exp_label = "test",
    datanum = 2000000,
    lr = 0.0003,
    batch_size = 256,
    beta = 5.0,
    time_rescale = False,
    multi_stage_genration = 2,

    epoch_offset=40,
    update_energy_end_epoch = 200,
    update_behavior_end_epoch = 200,
    update_flow_start_epoch = 0,
    flow_constrained_rl4_multiple_actions = 10,
    flow_constrained_rl5_multiple_actions = 10,
    adv_rl_multiple_actions = 30,

    isql_alpha = 3.0,
    isql_sofrmax_action_num = 32,

    x_record=False,
    flow_step_scale = 1.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_action, WeightedSamplesType.subopt2opt],

    gfpo_expectile=0.85,
)


#todo #############################  guided_flow ############################################
distributional_rl_flow_parameters = dict(
    # todo       halfcheetah    hopper     walker2d               random   medium  expert   medium-expert   medium-replay  full-replay                  v2
    # todo       antmaze        umaze  umaze-diverse medium-play medium-diverse large-play large-diverse         v1
    # todo       maze2d         umaze medium large umaze-dense medium-dense large-dense                          v0 v1
    debug_mode=True,
    # todo    dataset setting
    domain="mujoco",
    dataset='walker2d-medium-expert-v2',
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
    flow_step = 10,
    # todo training
    flow_guided_mode = FlowGuidedMode.normal,
    critic_type = CriticType.ciql,
    # flow_policy_type = FlowPolicyType.deterministic,
    rl_mode = RLTrainMode.use_rl_q,
    flow_value_TwinQ = False,
    large_q0_model=False,
    iql_tau = 0.5,
    expectile_func_tau = 0.9,
    expectile_flow_tau = 0.95,
    noise_scale_threshold = 0.1,
    noise_similarity_threshold = 0.85,
    divergence_coef = 1.0,
    conservative_coef = 1.0,
    divergence_discount = 1.0,
    weight_policy_regression_coef = 0.1,
    grpo_group_size = 10,
    grpo_exploration_rate = None,
    x_t_clip_value = 30.0,
    ppo_clip_rate = 0.2,
    ema_decay = 0.995,
    save_path = './output',
    load_critic_model = 'ciql_rl_flow',
    save_freq = 50000,
    energy_scale = 1,
    wandb_log = False,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = None,
    current_exp_label = "test",
    datanum = 2000000,
    lr = 0.0003,
    batch_size = 512,
    beta = 5.0,
    time_rescale = False,
    multi_stage_genration = 2,

    update_energy_end_epoch = 40,
    update_behavior_end_epoch = 40,
    update_flow_start_epoch = 40,

    x_record=False,
    flow_step_scale = 1.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_action, WeightedSamplesType.subopt2opt]
)

#todo #############################  flow q-learning ############################################
flow_q_learning_parameters = dict(
    # todo       halfcheetah    hopper     walker2d               random   medium  expert   medium-expert   medium-replay  full-replay                  v2
    # todo       antmaze        umaze  umaze-diverse medium-play medium-diverse large-play large-diverse         v1
    # todo       maze2d         umaze medium large umaze-dense medium-dense large-dense                          v0 v1
    debug_mode=True,
    # todo    dataset setting
    domain="mujoco",
    dataset='halfcheetah-medium-v2',
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
    flow_step = 10,
    direct_flow_step = 2,
    # todo training
    flow_guided_mode = FlowGuidedMode.normal,
    critic_type = CriticType.ciql,
    rl_mode = RLTrainMode.direct_flow_2_result,
    flow_value_TwinQ = False,
    large_q0_model=False,
    iql_tau = 0.5,
    expectile_func_tau = 0.9,
    expectile_flow_tau = 0.95,
    noise_scale_threshold = 0.1,
    noise_similarity_threshold = 0.85,
    divergence_coef = 1.0,
    conservative_coef = 1.0,
    divergence_discount = 1.0,
    weight_policy_regression_coef = 0.1,
    x_t_clip_value = 30.0,
    ema_decay = 0.995,
    save_path = './output',
    load_critic_model = 'ciql_rl_flow',
    save_freq = 50000,
    energy_scale = 1,
    wandb_log = False,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = None,
    current_exp_label = "test",
    datanum = 2000000,
    lr = 0.0003,
    batch_size = 1024,
    beta = 5.0,
    time_rescale = False,
    multi_stage_genration = 2,

    x_record=False,
    flow_step_scale = 1.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_action, WeightedSamplesType.subopt2opt]
)


adaptive_flow_step_parameters = dict(
    # todo       halfcheetah    hopper     walker2d               random   medium  expert   medium-expert   medium-replay  full-replay                  v2
    # todo       antmaze        umaze  umaze-diverse medium-play medium-diverse large-play large-diverse         v1
    # todo       maze2d         umaze medium large umaze-dense medium-dense large-dense                          v0 v1
    locomotion_dataset = ["halfcheetah-full-replay-v2", "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
                          "hopper-full-replay-v2", "hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2",
                          "walker2d-full-replay-v2", "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2"],
    adroit_dataset = ["hammer-expert-v1", "hammer-human-v1", "hammer-cloned-v1", "pen-expert-v1", "pen-human-v1", "pen-cloned-v1",
                      "relocate-expert-v1", "relocate-human-v1", "relocate-cloned-v1", "door-expert-v1", "door-human-v1", "door-cloned-v1"],
    maze2d_dataset = ["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1", "maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"],
    antmaze_dataset = ["antmaze-umaze-v1", "antmaze-umaze-diverse-v1", "antmaze-medium-play-v1", "antmaze-medium-diverse-v1", "antmaze-large-play-v1", "antmaze-large-diverse-v1"],
    debug_mode=False,
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
    flow_step = 10,
    eval_flow_step = 20,
    direct_flow_step = 1,
    multi_mode_action_evaluation=False,
    # todo training
    flow_guided_mode = FlowGuidedMode.normal,
    critic_type = CriticType.iql,
    rl_mode = RLTrainMode.adaptive_step,
    flow_value_TwinQ = False,
    large_q0_model=False,
    large_flow=False,
    large_flow_V=False,
    iql_tau = 0.5,
    expectile_func_tau = 0.9,
    expectile_flow_tau = 0.95,
    noise_scale_threshold = 0.1,
    noise_similarity_threshold = 0.85,
    divergence_coef = 1.0,
    conservative_coef = 1.0,
    divergence_discount = 1.0,
    weight_policy_regression_coef = 0.1,
    x_t_clip_value = 30.0,
    ema_decay = 0.995,
    save_path = './output',
    load_critic_model = 'ciql_rl_flow',
    save_freq = 50000,
    energy_scale = 1,
    wandb_log = True,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = None,
    current_exp_label = "test",
    datanum = 2000000,
    lr = 0.0003,
    batch_size = 256,
    beta = 5.0,
    time_rescale = False,
    multi_stage_genration = 2,

    epoch_offset=40,
    update_energy_end_epoch = 200,
    update_behavior_end_epoch = 200,
    update_flow_start_epoch = 0,
    flow_constrained_rl4_multiple_actions = 10,
    flow_constrained_rl5_multiple_actions = 10,
    adv_rl_multiple_actions = 30,

    isql_alpha = 3.0,
    isql_sofrmax_action_num = 32,

    x_record=False,
    flow_step_scale = 10.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_action, WeightedSamplesType.subopt2opt],

    gfpo_expectile=0.85,
)

distributional_flow_value_parameters = dict(
    # todo       halfcheetah    hopper     walker2d               random   medium  expert   medium-expert   medium-replay  full-replay                  v2
    # todo       antmaze        umaze  umaze-diverse medium-play medium-diverse large-play large-diverse         v1
    # todo       maze2d         umaze medium large umaze-dense medium-dense large-dense                          v0 v1
    locomotion_dataset = ["halfcheetah-full-replay-v2", "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
                          "hopper-full-replay-v2", "hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2",
                          "walker2d-full-replay-v2", "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2"],
    adroit_dataset = ["hammer-expert-v1", "hammer-human-v1", "hammer-cloned-v1", "pen-expert-v1", "pen-human-v1", "pen-cloned-v1",
                      "relocate-expert-v1", "relocate-human-v1", "relocate-cloned-v1", "door-expert-v1", "door-human-v1", "door-cloned-v1"],
    maze2d_dataset = ["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1", "maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"],
    antmaze_dataset = ["antmaze-umaze-v1", "antmaze-umaze-diverse-v1", "antmaze-medium-play-v1", "antmaze-medium-diverse-v1", "antmaze-large-play-v1", "antmaze-large-diverse-v1"],
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
    flow_step = 10,
    eval_flow_step = 20,
    direct_flow_step = 1,
    multi_mode_action_evaluation=False,
    # todo training
    flow_guided_mode = FlowGuidedMode.normal,
    critic_type = CriticType.iql,
    rl_mode = RLTrainMode.adaptive_step,
    flow_value_TwinQ = False,
    large_q0_model=False,
    large_flow=False,
    large_flow_V=False,
    iql_tau = 0.5,
    expectile_func_tau = 0.9,
    expectile_flow_tau = 0.95,
    noise_scale_threshold = 0.1,
    noise_similarity_threshold = 0.85,
    divergence_coef = 1.0,
    conservative_coef = 1.0,
    divergence_discount = 1.0,
    weight_policy_regression_coef = 0.1,
    x_t_clip_value = 500.0,
    ema_decay = 0.995,
    save_path = './output',
    load_critic_model = 'ciql_rl_flow',
    save_freq = 50000,
    energy_scale = 1,
    wandb_log = False,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = None,
    current_exp_label = "test",
    datanum = 2000000,
    lr = 0.0003,
    batch_size = 256,
    beta = 5.0,
    time_rescale = False,
    multi_stage_genration = 2,

    epoch_offset=40,
    update_energy_end_epoch = 200,
    update_behavior_end_epoch = 200,
    update_flow_start_epoch = 0,
    flow_constrained_rl4_multiple_actions = 10,
    flow_constrained_rl5_multiple_actions = 10,
    adv_rl_multiple_actions = 30,

    isql_alpha = 3.0,
    isql_sofrmax_action_num = 32,

    x_record=False,
    flow_step_scale = 10.0,
    color_list=["#ff595e", "#8ac926", "#1982c4", "#6a4c93"],
    weighted_samples_type = [WeightedSamplesType.noise_action, WeightedSamplesType.subopt2opt],

    gfpo_expectile=0.85,
)
