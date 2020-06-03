"""
Script to simulate 1-dimensional dynamical systems used in Activation Dynamic Networks (ADN) for different hyper-parameters
"""
import numpy as np
import torch as to
import os.path as osp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.adn import ADNPolicy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Load the environment and the policy
    env, policy, _ = load_experiment(ex_dir, args)
    if not isinstance(policy, ADNPolicy):
        raise pyrado.TypeErr(given=policy, expected_type=ADNPolicy)

    # Define the parameters for evaluation
    num_steps = 1000
    p_init_min, p_init_max, num_p_init = -6., 6., 11
    print_cbt(f'Evaluating an ADNPolicy for {num_steps} steps ad {1/env.dt} Hz with initial potentials ranging from'
              f'{p_init_min} to {p_init_max}.', 'c')

    time = to.linspace(0., num_steps*env.dt, num_steps)  # endpoint included
    p_init = to.linspace(p_init_min, p_init_max, num_p_init)  # endpoint included
    num_p = len(policy.potentials)

    fig, ax = plt.subplots(1, figsize=(12, 10), subplot_kw={'projection': '3d'})
    fig.canvas.set_window_title('Potentials over time for various initial potentials')

    for p_0 in p_init:
        p = to.zeros(num_steps, num_p)
        s = to.zeros(num_steps, num_p)
        policy._potentials = p_0*to.ones_like(policy.potentials)

        for i in range(num_steps):
            p[i, :] = policy.potentials + env.dt*policy.potentials_dot(stimuli=s[i, :])
            policy._potentials = p[i, :].clone()

        # Plot
        plt.plot(time.numpy(), p_0.repeat(num_steps).numpy(), p[:, 0].detach().numpy())

    # Save
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(ex_dir, f'potdyn-kappa.{fmt}'), dpi=500)

    plt.show()
