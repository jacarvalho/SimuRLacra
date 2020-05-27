"""
Simulate (with animation) a policy for the WAM Ball-in-cup task.
Export the policy in form of desired joint position and velocities.
The converted policy is saved same directory where the original policy was loaded from.
"""
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import pyrado
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.sampling.rollout import rollout
from pyrado.utils.input_output import print_cbt

if __name__ == '__main__':
    # Fix seed for reproducibility
    pyrado.set_seed(101)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Load the environment and the policy
    env, policy, kwout = load_experiment(ex_dir)

    # env = remove_all_dr_wrappers(env)

    ro = rollout(env, policy, render_mode=RenderMode(video=True), eval=True)
    print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
    t = ro.env_infos['t']
    des_qpos = ro.env_infos['des_qpos']
    des_qvel = ro.env_infos['des_qvel']
    # np.concatenate([des_qpos, des_qvel], axis=1)
    np.save(osp.join(ex_dir, 'des_qpos.npy'), des_qpos)
    np.save(osp.join(ex_dir, 'des_qvel.npy'), des_qvel)

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(3, 2, sharex='all')
    fig.suptitle('Desired Trajectory')
    for i, idx in enumerate([1, 3, 5]):
        ax[i, 0].plot(t, des_qpos[:, idx])
        ax[i, 1].plot(t, des_qvel[:, idx])
        ax[i, 0].set_ylabel(f'joint {idx}')
    ax[2, 0].set_xlabel('time [s]')
    ax[2, 1].set_xlabel('time [s]')
    ax[0, 0].set_title('joint pos')
    ax[0, 1].set_title('joint vel')
    plt.show()