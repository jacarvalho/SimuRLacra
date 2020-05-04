import numpy as np

from pyrado.environment_wrappers.base import EnvWrapperAct
from pyrado.spaces.box import BoxSpace


class ActNormWrapper(EnvWrapperAct):
    """ Environment wrapper which normalizes the action space, such that all action values are in range [-1, 1]. """

    def _process_act(self, act: np.ndarray) -> np.ndarray:
        # Get the bounds of the inner action space
        lb, ub = self.wrapped_env.act_space.bounds

        # Denormalize action
        act_denorm = lb + (act + 1) * (ub - lb) / 2

        return act_denorm  # can be out of action space, but this has to be checked by the environment

    def _process_act_space(self, space: BoxSpace) -> BoxSpace:
        # New bounds are [-1, 1]
        ub = np.ones(space.shape)
        return BoxSpace(-ub, ub, labels=space.labels)
