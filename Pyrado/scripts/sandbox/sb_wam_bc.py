"""
Test PD controller for stabilizing the WAM at its init position.
"""
import numpy as np

from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.utils.data_types import RenderMode


if __name__ == '__main__':
    # Define the gains and limits for the controller
    p_gains = np.array([200, 300, 100, 100, 10, 10, 2.5])
    d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

    # Environment
    env = WAMBallInCupSim()
    obs = env.reset()
    des_state = obs.copy()
    env.render(RenderMode(video=True))
    env.viewer._paused = True

    for _ in range(5000):
        # PD controller (no gravity compensation)
        # act = p_gains * (des_state[:7] - obs[:7]) + d_gains * (des_state[7:] - obs[7:])

        # Zero action
        # act = np.zeros_like(env.act_space.sample_uniform())

        # Gravity compensation: http://www.mujoco.org/forum/index.php?threads/gravitational-matrix-calculation.3404/
        act = env.sim.data.qfrc_bias[:7]
        obs, _, _, _ = env.step(act)
        env.render(RenderMode(video=True))
