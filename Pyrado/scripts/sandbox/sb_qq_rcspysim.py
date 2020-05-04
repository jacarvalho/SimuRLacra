"""
This is a very basic script to test the functionality of Rcs & RcsPySim & Pyrado using the Quanser Qube setup.
"""
import numpy as np

from pyrado.environments.rcspysim.quanser_qube import QQubeRcsSim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.plotting.rollout_based import plot_observations_actions_rewards
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


if __name__ == '__main__':
    # Set up environment
    dt = 1/5000.
    max_steps = 5000
    env = QQubeRcsSim(
        physicsEngine='Bullet',  # Bullet or Vortex
        dt=dt,
        max_steps=max_steps,
        max_dist_force=None
    )
    print_domain_params(env.domain_param)

    # Set up policy
    policy = TimePolicy(env.spec, lambda t: [1.], dt)  # constant acceleration with 1. rad/s**2

    # Simulate
    ro = rollout(
        env, policy, render_mode=RenderMode(video=True),
        reset_kwargs=dict(init_state=np.array([0, 3/180*np.pi, 0, 0.]))
    )

    # Plot
    print(f'After {max_steps*dt} s of accelerating with 1. rad/s**2, we should be at {max_steps*dt} rad/s')
    print(f'Difference: {max_steps*dt - ro.observations[-1][2]} rad/s (mind the swinging pendulum)')
    plot_observations_actions_rewards(ro)
