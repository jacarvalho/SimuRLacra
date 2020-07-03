"""
Plot trajectory recorded on real Barrett WAM and compare it to the simulation.
"""
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import pyrado
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt

if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Do a simple loop to ask again for a directory if the chosen one does not contain a real-world trajectory
    data_exists = False
    while not data_exists:
        # Get the experiment's directory to load from if not given as command line argument
        ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

        # Load real trajectories
        try:
            qpos_real = np.load(osp.join(ex_dir, 'qpos_real.npy'))
            qvel_real = np.load(osp.join(ex_dir, 'qvel_real.npy'))
            data_exists = True
        except FileNotFoundError:
            print_cbt('Real trajectory has not been recorded for this policy!', 'r', bright=True)

    # Load the policy and the environment
    env, policy, _ = load_experiment(ex_dir, args)
    # Get nominal environment
    env = remove_all_dr_wrappers(env)
    env.domain_param = env.get_nominal_domain_param()

    # Fix seed for reproducibility
    pyrado.set_seed(args.seed)

    # Do rollout in simulation
    init_state = env.init_space.sample_uniform()
    # Use same initial state as on real system
    # init_state[:7] = qpos_real[0]
    ro = rollout(env, policy, eval=True, render_mode=RenderMode(video=False), reset_kwargs=dict(init_state=init_state))
    t, qpos_sim, qvel_sim = ro.env_infos['t'], ro.env_infos['qpos'], ro.env_infos['qvel']

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(3, 2, sharex='all')
    fig.suptitle('Trajectory comparison')
    for i, idx in enumerate([1, 3, 5]):
        ax[i, 0].plot(t, qpos_sim[:, idx], label='Sim')
        ax[i, 0].plot(t, qpos_real[:, idx], label='Real')
        ax[i, 1].plot(t, qvel_sim[:, idx], label='Sim')
        ax[i, 1].plot(t, qvel_real[:, idx], label='Real')
        ax[i, 0].set_ylabel(f'joint {idx}')
        if i == 0:
            ax[i, 0].legend()
            ax[i, 1].legend()
    ax[2, 0].set_xlabel('time [s]')
    ax[2, 1].set_xlabel('time [s]')
    ax[0, 0].set_title('joint pos [rad]')
    ax[0, 1].set_title('joint vel [rad/s]')
    plt.show()