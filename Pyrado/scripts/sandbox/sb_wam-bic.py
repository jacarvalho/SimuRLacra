"""
Planned: Test Linear Policy with RBF Features for the WAM Ball-in-a-cup task.
Current: Stabilize WAM arm at his initial position
"""
import numpy as np

from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.utils.data_types import RenderMode
from pyrado.policies.features import FeatureStack, RBFFeat
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Environment
    env = WAMBallInCupSim()
    env.reset()
    env.render(mode=RenderMode(video=True))
    env.viewer._paused = True

    # Zero action; corresponds to holding the WAM at his initial position
    act = np.zeros((6,))
    for i in range(3000):
        env.step(act)
        env.render(mode=RenderMode(video=True))
