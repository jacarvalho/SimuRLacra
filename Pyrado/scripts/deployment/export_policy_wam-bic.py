"""
Simulate (with animation) a policy for the WAM Ball-in-cup task.
Export the policy in form of desired joint position and velocities.
The converted policy is saved same directory where the original policy was loaded from.
"""
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import pyrado
from pyrado.domain_randomization.utils import print_domain_params
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

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load the policy and the environment (for constructing the real-world counterpart)
    env, policy, _ = load_experiment(ex_dir, args)
    env = remove_all_dr_wrappers(env)
    env.domain_param = env.get_nominal_domain_param()
    print_cbt(f'Set up the environment with dt={env.dt} max_steps={env.max_steps}.', 'c')
    print_domain_params(env.domain_param)

    # Get the initial state from the command line, if given. Else, set None to delegate to the environment.
    if args.init_state is not None:
        init_state = env.init_space.sample_uniform()
        init_qpos = np.asarray(args.init_state)
        assert len(init_qpos) == 5
        np.put(init_state, [1, 3, 5, 6, 7], init_qpos)  # the passed init state only concerns certain joint angles
    else:
        init_state = None

    # Fix seed for reproducibility
    pyrado.set_seed(args.seed)

    # Do the rollout
    ro = rollout(env, policy, eval=True, render_mode=RenderMode(video=args.animation),
                 reset_kwargs=dict(init_state=init_state))

    # Save the trajectories
    if not hasattr(ro, 'env_infos'):
        raise KeyError('Rollout does not have the field env_infos!')
    t = ro.env_infos['t']
    qpos, qvel = ro.env_infos['qpos'], ro.env_infos['qvel']
    qpos_des, qvel_des = ro.env_infos['qpos_des'], ro.env_infos['qvel_des']
    np.save(osp.join(ex_dir, 'qpos.npy'), qpos)
    np.save(osp.join(ex_dir, 'qvel.npy'), qvel)
    np.save(osp.join(ex_dir, 'qpos_des.npy'), qpos_des)
    np.save(osp.join(ex_dir, 'qvel_des.npy'), qvel_des)

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(3, 2, sharex='all', constrained_layout=True)
    fig.suptitle('Desired Trajectory')
    for i, idx in enumerate([1, 3, 5]):
        ax[i, 0].plot(t, 180/np.pi*qpos_des[:, idx], label='Desired')
        ax[i, 0].plot(t, 180/np.pi*qpos[:, idx], label='Actual')
        ax[i, 1].plot(t, 180/np.pi*qvel_des[:, idx], label='Desired')
        ax[i, 1].plot(t, 180/np.pi*qvel[:, idx], label='Actual')
        ax[i, 0].set_ylabel(f'joint {idx}')
        if i == 0:
            ax[i, 0].legend()
            ax[i, 1].legend()
    ax[2, 0].set_xlabel('time [s]')
    ax[2, 1].set_xlabel('time [s]')
    ax[0, 0].set_title('joint pos [deg]')
    ax[0, 1].set_title('joint vel [deg/s]')
    plt.show()
