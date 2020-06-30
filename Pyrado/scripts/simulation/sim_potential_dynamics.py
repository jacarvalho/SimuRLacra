"""
Script to simulate 1-dimensional dynamical systems used in Activation Dynamic Networks (ADN) for different hyper-parameters
"""
import numpy as np
import torch as to
import os.path as osp
from matplotlib import pyplot as plt

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.adn import ADNPolicy
from pyrado.policies.neural_fields import NFPolicy
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
    if not isinstance(policy, (ADNPolicy, NFPolicy)):
        raise pyrado.TypeErr(given=policy, expected_type=[ADNPolicy, NFPolicy])

    # Define the parameters for evaluation
    num_steps, dt_eval = 200, env.dt/10
    policy._dt = dt_eval
    p_init_min, p_init_max, num_p_init = -6., 6., 11
    print_cbt(f'Evaluating an ADNPolicy for {num_steps} steps ad {1/dt_eval} Hz with initial potentials ranging from'
              f'{p_init_min} to {p_init_max}.', 'c')

    time = to.linspace(0., num_steps*dt_eval, num_steps)  # endpoint included
    p_init = to.linspace(p_init_min, p_init_max, num_p_init)  # endpoint included
    num_p = len(policy.potentials)

    # Iterate over all action dimensions since we got as many potential-based neurons as action dimensions
    # For mode = standalone they are currently all the same because all neuron potential-based obey the same dynamics.
    # However, this does not necessarily have to be that way. Thus we plot the same way as for mode = policy.
    for a in range(policy.hidden_size):

        fig, ax = plt.subplots(1, figsize=(12, 10), subplot_kw={'projection': '3d'})
        fig.canvas.set_window_title(f'Potential dynamics for the {a}-th action dimension for initial values')
        ax.set_xlabel('$t$ [s]')
        ax.set_ylabel('$p_0$')
        ax.set_zlabel('$p(t)$')

        for p_0 in p_init:
            p = to.zeros(num_steps, num_p)
            s = to.zeros(num_steps, num_p)
            policy._potentials = p_0*to.ones_like(policy.potentials)
            hidden = policy.init_hidden()

            for i in range(num_steps):
                if args.mode == 'standalone':
                    # Use the potential dynamics as defined in the loaded ADNPolicy
                    p[i, :] = policy.potentials + dt_eval*policy.potentials_dot(stimuli=s[i, :])
                    policy._potentials = p[i, :].clone()

                elif args.mode == 'policy':
                    # Use the loaded ADNPolicy's forward method and pass zero-observations
                    p[i, :] = policy._potentials.clone()
                    _, hidden = policy(to.zeros(policy.env_spec.obs_space.shape), hidden)  # previous action is in hidden

                else:
                    raise pyrado.ValueErr(given=args.mode, given_name='mode', eq_constraint="'standalone' or 'policy'")

            # Plot
            plt.plot(time.numpy(), p_0.repeat(num_steps).numpy(), p[:, a].detach().numpy())
        plt.title(f'Final value {p[-1, a].numpy().round(4)}',  y=1.05)

    # Save
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(ex_dir, f'potdyn-kappa.{fmt}'), dpi=500)

    plt.show()
