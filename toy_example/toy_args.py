import argparse

def bandit_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="8gaussians") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str)    #
    parser.add_argument("--device", default="cuda", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=3.0)        # beta parameter in the paper, use alpha because of legacy
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--method', type=str, default="CEP")
    print("**************************")
    args = parser.parse_known_args()[0]
    print(args)
    return args