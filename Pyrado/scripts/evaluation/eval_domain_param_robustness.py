"""
Evaluate robustness of policies towards changes in single domain parameters
"""
import numpy as np
import os
import torch as to

import pyrado
from matplotlib import pyplot as plt
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout
from pyrado import set_seed
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    if args.max_steps == pyrado.inf:
        args.max_steps = 600
        print_cbt(f'Set maximum number of time steps to {args.max_steps}', 'y')

    # ex_dir = input('Enter a root directory that contains one or more experiment directories:\n')
    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()
    dirs = [x[0] for x in os.walk(ex_dir)][1:]
    num_policies = len(dirs)
    print(f'Found {num_policies} policies.')

    # Specify domain parameters
    param_names = ['Dp', 'Dr', 'Mp', 'Mr', 'Lp', 'Lr']
    num_param = len(param_names)
    num_samples = 10

    # Create one-dim evaluation grid for multiple parameters
    nom_params = QQubeSim.get_nominal_domain_param()
    param_values = dict(
            Dp=np.logspace(-8, -4, num_samples),
            Dr=np.logspace(-8, -4, num_samples),
            Mp=np.linspace(0.6 * nom_params['Mp'], 1.5 * nom_params['Mp'], num_samples),
            Mr=np.linspace(0.6 * nom_params['Mr'], 1.5 * nom_params['Mr'], num_samples),
            Lp=np.linspace(0.6 * nom_params['Lp'], 1.5 * nom_params['Lp'], num_samples),
            Lr=np.linspace(0.6 * nom_params['Lr'], 1.5 * nom_params['Lr'], num_samples),
    )

    # Set up the environment
    env = ActNormWrapper(QQubeSim(dt=1 / 100., max_steps=args.max_steps))

    # Evaluate the performance of each policy on the specified domain parameter ranges
    rewards = np.empty([num_policies, num_param, num_samples])

    for i, pol_dir in enumerate(dirs):
        print(f'Evaluating policy {i + 1} of {num_policies} ...')
        policy = to.load(os.path.join(pol_dir, 'policy.pt'))

        # Synchronize seeds between policies
        set_seed(args.seed + i)

        # Loop over domain parameters
        for j, name in enumerate(param_names):

            # Loop over domain parameter values
            for k, value in enumerate(param_values[name]):
                # Get reward
                rewards[i, j, k] = rollout(env, policy, eval=True,
                                           reset_kwargs=dict(domain_param={name: value})
                                           ).undiscounted_return()

    ''' Plotting '''
    plt.figure()

    for i, name in enumerate(param_names):
        plt.subplot(3, 2, i + 1)
        plt.plot(param_values[name], np.mean(rewards[:, i], axis=0), c='k')

        for j in range(num_policies):
            plt.scatter(param_values[name], rewards[j, i, :], marker='o')

        plt.fill_between(param_values[name], rewards[:, i].min(axis=0), rewards[:, i].max(axis=0),
                         facecolor='b', edgecolor=None, alpha=0.3)

        if name == 'Dp' or name == 'Dr':
            plt.xscale('log')

        plt.ylim(0, args.max_steps)
        plt.ylabel('Reward')
        plt.xlabel(name)

    plt.show()
