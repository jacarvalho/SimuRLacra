"""
Script to generate time series data sets.
"""
import argparse
import functools
import numpy as np
import os.path as osp
from typing import Callable

import pyrado
from matplotlib import pyplot as plt
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.dummy import IdlePolicy
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout


def _dirac_impulse(t, env_spec, amp) -> Callable:
    return amp * np.ones(env_spec.act_space.shape) if t == 0 else np.zeros(env_spec.act_space.shape)


def generate_oscillation_data(dt, t_end, excitation):
    """
    Use OMOEnv to generate a 1-dim damped oscillation signal.

    :param dt: time step size [s]
    :param t_end: Time duration [s]
    :param excitation: type of excitation, either (initial) 'position' or 'force' (function of time)
    :return: 1-dim oscillation trajectory
    """
    env = OneMassOscillatorSim(dt, np.ceil(t_end / dt))
    env.domain_param = dict(m=1., k=10., d=2.0)
    if excitation == 'force':
        policy = TimePolicy(env.spec, functools.partial(_dirac_impulse, env_spec=env.spec, amp=0.5), dt)
        reset_kwargs = dict(init_state=np.array([0, 0]))
    elif excitation == 'position':
        policy = IdlePolicy(env.spec)
        reset_kwargs = dict(init_state=np.array([0.5, 0]))
    else:
        raise pyrado.ValueErr(given=excitation, eq_constraint="'force' or 'position'")

    # Generate the data
    ro = rollout(env, policy, reset_kwargs=reset_kwargs, record_dts=False)
    return ro.observations[:, 0]


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('excitation', type=str, nargs='?', default='position',
                        help="Type of excitation ('position' or 'force')")
    parser.add_argument('dt', type=float, nargs='?', default='0.01', help='Time step size [s]')
    parser.add_argument('t_end', type=float, nargs='?', default='5.', help='Time duration [s]')
    parser.add_argument('std', type=float, nargs='?', default='0.02', help='Standard deviation of the noise')
    args = parser.parse_args()

    # Generate ground truth data
    data_gt = generate_oscillation_data(0.01, 5, args.excitation)

    # Add noise
    noise = np.random.randn(*data_gt.shape) * args.std
    data_n = data_gt + noise

    # Plot the data
    plt.plot(data_n, label='signal')
    plt.plot(data_gt, lw=3, label='ground truth')
    plt.legend()
    plt.show()

    # Save the data
    np.save(osp.join(pyrado.PERMA_DIR, 'time_series', 'omo_traj_gt.npy'), data_gt)
    np.save(osp.join(pyrado.PERMA_DIR, 'time_series', 'omo_traj_n.npy'), data_n)
