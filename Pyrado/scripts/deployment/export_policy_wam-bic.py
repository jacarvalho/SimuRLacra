"""
Simulate (with animation) a policy for the WAM Ball-in-cup task.
Export the policy in form of desired joint position and velocities.
The converted policy is saved same directory where the original policy was loaded from.
"""
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import pyrado
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment, wrap_like_other_env
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load the policy and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir)

    # Create real-world counterpart (without domain randomization)
    env_real = WAMBallInCupSim(max_steps=env_sim.max_steps)  # TODO @Christian: we way want to implement a WAMBallInCupReal class that runs RobCom (without mujoco-py)
    print_cbt(f'Set up the env_real environment with dt={env_real.dt} max_steps={env_real.max_steps}.', 'c')
    env_real = wrap_like_other_env(env_real, env_sim)

    # Get the initial state from the command line, if given. Else, set None to delegate to the environment.
    if args.init_state is not None:
        init_state = env_sim.init_space.sample_uniform()
        init_qpos = np.asarray(args.init_state)
        assert len(init_qpos) == 5
        np.put(init_state, [1, 3, 5, 6, 7], init_qpos)  # the passed init state only concerns certain joint angles
    else:
        init_state = None

    # Fix seed for reproducibility
    pyrado.set_seed(args.seed)

    # Do the rollout and save the trajectories
    ro = rollout(env_real, policy, eval=True, render_mode=RenderMode(video=True),
                 reset_kwargs=dict(init_state=init_state))
    if not hasattr(ro, 'env_infos'):
        raise KeyError('Rollout does not have the field env_infos!')
    t, des_qpos, des_qvel = ro.env_infos['t'], ro.env_infos['des_qpos'], ro.env_infos['des_qvel']
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
    ax[0, 0].set_title('joint pos [rad]')
    ax[0, 1].set_title('joint vel [rad/s]')
    plt.show()
