"""
Test the downsampling wrapper.
"""
import numpy as np

from matplotlib import pyplot as plt
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.downsampling import DownsamplingWrapper
from pyrado.sampling.rollout import rollout
from pyrado.policies.environment_specific import QBallBalancerPDCtrl
from pyrado.utils.data_types import RenderMode


if __name__ == '__main__':
    # Set up environment
    factor = 5  # don't change this
    dt = 1/500.  # don't change this
    max_steps = 2000  # don't change this
    init_state = np.array([0, 0, 0.1, 0.1, 0, 0, 0, 0])
    env = QBallBalancerSim(dt=dt, max_steps=max_steps)
    env = ActNormWrapper(env)

    # Set up policy
    policy = QBallBalancerPDCtrl(env.spec)

    # Simulate
    ro = rollout(env, policy,
                 reset_kwargs=dict(domain_param=dict(dt=dt), init_state=init_state),
                 render_mode=RenderMode(video=True), max_steps=max_steps)
    act_500Hz = ro.actions

    ro = rollout(env, policy,
                 reset_kwargs=dict(domain_param=dict(dt=dt*factor), init_state=init_state),
                 render_mode=RenderMode(video=True), max_steps=int(max_steps/factor))
    act_100Hz = ro.actions
    act_100Hz_zoh = np.repeat(act_100Hz, 5, axis=0)

    env = DownsamplingWrapper(env, factor)
    ro = rollout(env, policy,
                 reset_kwargs=dict(domain_param=dict(dt=dt), init_state=init_state),
                 render_mode=RenderMode(video=True), max_steps=max_steps)
    act_500Hz_wrapped = ro.actions

    # Plot
    _, axs = plt.subplots(nrows=2)
    for i in range(2):
        axs[i].plot(act_500Hz[:, i], label='500 Hz (original)')
        axs[i].plot(act_100Hz_zoh[:, i], label='100 Hz (zoh)')
        axs[i].plot(act_500Hz_wrapped[:, i], label='500 Hz (wrapped)')
        axs[i].legend()
        axs[i].set_ylabel(env.act_space.labels[i])
    axs[1].set_xlabel('time steps')
    plt.show()
