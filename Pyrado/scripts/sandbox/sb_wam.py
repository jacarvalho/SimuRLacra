"""
Test PD controller for driving the WAM to a desired position.
"""
import numpy as np

from pyrado.environments.mujoco.wam import WAMSim
from pyrado.utils.data_types import RenderMode


if __name__ == '__main__':
    # Define the gains and limits for the controller
    p_gains = np.array([200, 300, 100, 100, 10, 10, 2.5])
    d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

    n = 1500  # Number of steps
    init_pos = np.array([0., -1.986, 0., 3.146, 0., 0., 0.])
    zero_vel = np.zeros_like(init_pos)
    goal_pos = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    # constants
    diff = goal_pos - init_pos
    c_1 = -2 * diff / n ** 3
    c_2 = 3 * diff / n ** 2

    # Environment
    env = WAMSim()
    env.reset(init_state=np.concatenate((init_pos, zero_vel)).ravel())
    env.render(mode=RenderMode(video=True))
    env.viewer._run_speed = 0.5

    for i in range(1, n+1000):
        if i < n:
            des_pos = c_1 * i**3 + c_2 * i**2 + init_pos
            des_vel = (3 * c_1 * i**2 + 2 * c_2 * i) / env.dt
        else:
            des_pos = goal_pos
            des_vel = zero_vel
        act = p_gains * (des_pos - env.state[:7]) + d_gains * (des_vel - env.state[7:])
        env.step(act)
        env.render(mode=RenderMode(video=True))

    print('Desired Pos:', goal_pos)
    print('Actual Pos:', env.state[:7])