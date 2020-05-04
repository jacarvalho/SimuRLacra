"""
Script to plot the PyTorch parameters of a policy
"""
import os.path as osp
from matplotlib import pyplot as plt

from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.policy_parameters import render_policy_params
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Load the policy
    env, policy, _ = load_experiment(ex_dir, args)

    # Print the policy structure
    print(policy)

    # Visualize the parameters
    fig = render_policy_params(policy, env.spec, annotate=args.verbose)

    if args.save_figures:
        fig.savefig(osp.join(ex_dir, f'policy_nn_weights.pdf'), dpi=500)

    plt.show()
