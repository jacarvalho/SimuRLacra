import numpy as np
from typing import Sequence

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.tasks.base import Task
from pyrado.tasks.utils import never_succeeded
from pyrado.tasks.reward_functions import RewFcn


class EndlessFlippingTask(Task):
    """
    Task class for flipping an object around one axis about a desired angle. Once the new angle is equal to the
    old angle plus/minus a given angle delta, the new angle becomes the old one and the flipping continues.
    """

    def __init__(self,
                 env_spec: EnvSpec,
                 rew_fcn: RewFcn,
                 init_angle: float,
                 des_angle_delta: float = np.pi/2.,
                 angle_tol: float = 1/180.*np.pi):
        """
        Constructor

        :param env_spec: environment specification of a simulated or real environment
        :param rew_fcn: reward function, an instance of a subclass of RewFcn
        :param init_angle: initial angle
        :param des_angle_delta: desired angle that counts as a flip
        :param angle_tol: tolerance
        """
        if not isinstance(env_spec, EnvSpec):
            raise pyrado.TypeErr(given=env_spec, expected_type=EnvSpec)
        if not isinstance(rew_fcn, RewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=RewFcn)

        self._env_spec = env_spec
        self._rew_fcn = rew_fcn
        self._init_angle = init_angle
        self._last_angle = init_angle
        self.des_angle_delta = des_angle_delta
        self.angle_tol = angle_tol
        self._held_rew = 0.

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def rew_fcn(self) -> RewFcn:
        return self._rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: RewFcn):
        if not isinstance(rew_fcn, RewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=RewFcn)
        self._rew_fcn = rew_fcn

    def reset(self, env_spec: EnvSpec, init_angle: float = None, **kwargs):
        """
        Reset the task.

        :param env_spec: environment specification
        :param init_angle: override initial angle
        :param kwargs: keyword arguments forwarded to the reward function, e.g. the initial state
        """
        # Update the environment specification at every reset of the environment since the spaces could change
        self._env_spec = env_spec

        # Reset the internal quantities to recognize the flips
        self._last_angle = init_angle if init_angle is not None else self._init_angle
        self._held_rew = 0.

        # Some reward functions scale with the state and action bounds
        self._rew_fcn.reset(state_space=env_spec.state_space, act_space=env_spec.act_space, **kwargs)

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # We don't care about the flip direction or the number of revolutions.
        des_angles_both = np.array([[self._last_angle + self.des_angle_delta],
                                    [self._last_angle - self.des_angle_delta]])
        err_state = des_angles_both - state
        err_state = np.fmod(err_state, 2*np.pi)  # map to [-2pi, 2pi]

        # Choose the closer angle for the reward. Operate on state and action errors
        rew = self._held_rew + self._rew_fcn(np.min(err_state, axis=0), -act, remaining_steps)  # act_des = 0

        # Check if the flip was successful
        succ_idx = abs(err_state) <= self.angle_tol
        if any(succ_idx):
            # If successful, increase the permanent reward and memorize the achieved goal angle
            self._last_angle = float(des_angles_both[succ_idx])
            self._held_rew += self._rew_fcn(np.min(err_state, axis=0), -act, remaining_steps)

        return rew

    def has_succeeded(self, state: np.ndarray) -> bool:
        return never_succeeded()
