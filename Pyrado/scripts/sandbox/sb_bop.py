"""
Script to test the functionality of Rcs & RcsPySim & Pyrado using a robotic ball-on-plate setup
"""
import math
from matplotlib import pyplot as plt

import rcsenv
import pyrado
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate5DSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt

rcsenv.setLogLevel(4)


def create_setup(physics_engine, dt, max_steps, max_dist_force):
    # Set up environment
    env = BallOnPlate5DSim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force
    )
    env = ActNormWrapper(env)
    print_domain_params(env.domain_param)

    # Set up policy
    def policy_fcn(t: float):
        return [
            0.0,  # x_ddot_plate
            0.5*math.sin(2.*math.pi*5*t),  # y_ddot_plate
            5.*math.cos(2.*math.pi/5.*t),  # z_ddot_plate
            0.0,  # alpha_ddot_plate
            0.0,  # beta_ddot_plate
        ]
    policy = TimePolicy(env.spec, policy_fcn, dt)

    return env, policy


if __name__ == '__main__':
    # Initialize
    fig, axs = plt.subplots(3, figsize=(8, 12), sharex='col', tight_layout=True)

    # Try to run several possible cases
    for pe in ['Bullet', 'Vortex']:
        print_cbt(f'Running with {pe} physics engine', 'c')

        if rcsenv.supportsPhysicsEngine(pe):
            env, policy = create_setup(pe, dt=0.01, max_steps=1000, max_dist_force=0.)

            # Simulate
            pyrado.set_seed(1)
            ro = rollout(env, policy, render_mode=RenderMode(video=True))

            # Render plots
            axs[0].plot(ro.observations[:, 0], label=pe)
            axs[1].plot(ro.observations[:, 1], label=pe)
            axs[2].plot(ro.observations[:, 2], label=pe)
            axs[0].legend()
            axs[1].legend()
            axs[2].legend()

    # Show plots
    axs[0].set_title('gBotKuka.xml')
    axs[0].set_ylabel('plate x pos')
    axs[1].set_ylabel('plate y pos')
    axs[2].set_ylabel('plate z pos')
    axs[2].set_xlabel('time steps')
    plt.show()
