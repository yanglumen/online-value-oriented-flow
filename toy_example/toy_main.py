import scipy
from toy_example.toy_dataset import *
from bandit_toy import *
from matplotlib import pyplot as plt
plt.rc('font', family='Times New Roman',size=16)
args = bandit_get_args()
marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
score_model= Bandit_MlpScoreNet(input_dim=2, output_dim=2, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)

def energy_sample(env, ss, sample_per_state=1000, **kwargs):
    data, e = inf_train_gen(env,batch_size=1000*sample_per_state)
    ori_e = e
    e = e * ss

    index = np.random.choice(1000*sample_per_state, p=scipy.special.softmax(e).squeeze(), size=sample_per_state, replace=False)
    data = data[index]
    ori_e = ori_e[index]
    return data, ori_e

def plot_ground_truth():
    for task in ["swissroll"]:
        plt.figure(figsize=(12, 3.0))
        ss = [0, 3, 10, 20]
        axes = []
        for i, s in enumerate(ss):
            plt.subplot(1, len(ss), i + 1)
            data, e = energy_sample(task, s, sample_per_state=2000)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(-4.5, 4.5)
            plt.ylim(-4.5, 4.5)
            if i == 0:
                mappable = plt.scatter(data[:, 0], data[:, 1], s=1, c=e, cmap="winter", vmin=0, vmax=1, rasterized=True)
                plt.yticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
            else:
                plt.scatter(data[:, 0], data[:, 1], s=1, c=e, cmap="winter", vmin=0, vmax=1, rasterized=True)
                plt.yticks(ticks=[-4, -2, 0, 2, 4], labels=[None, None, None, None, None])
            axes.append(plt.gca())
            plt.xticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
            plt.title(r'$\beta={}$'.format(s))
        plt.tight_layout()
        plt.gcf().colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
        plt.show()

if "__main__" == __name__:
    plot_ground_truth()