"""
This is a very basic script to test the functionality of Rcs & RcsPySim & Pyrado using a robotic ball-on-plate setup
using an untrained recurrent policy.
"""
import torch as to

import rcsenv
from matplotlib import pyplot as plt
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate5DSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.rnn import LSTMPolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode

rcsenv.setLogLevel(4)

if __name__ == '__main__':
    # Set up environment
    dt = 0.01
    env = BallOnPlate5DSim(
        physicsEngine='Vortex',  # Bullet or Vortex
        dt=dt,
        max_steps=2000,
    )
    env = ActNormWrapper(env)
    print_domain_params(env.domain_param)

    # Set up policy
    policy = LSTMPolicy(env.spec, 20, 1)
    policy.init_param()

    # Simulate
    ro = rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex='all', tight_layout=True)
    axs[0].plot(ro.observations[:, 1], label='plate y pos')
    axs[1].plot(ro.observations[:, 2], label='plate z pos')
    axs[0].legend()
    axs[1].legend()
    plt.show()

    ro.torch()

    # Simulate in the policy's eval mode
    ev_act, _ = policy(ro.observations[:-1, ...], ro.hidden_states)
    print(f'Difference in actions between recordings in simulation and the policy evaluation mode:\n'
          f'{to.max(to.abs(ev_act - ro.actions))}')
