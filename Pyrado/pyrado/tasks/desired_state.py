import numpy as np
from typing import Sequence

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.tasks.base import Task
from pyrado.tasks.utils import never_succeeded
from pyrado.tasks.reward_functions import RewFcn


class DesStateTask(Task):
    """ Task class for moving to a (fixed) desired state. Operates on the error in state and action. """

    def __init__(self,
                 env_spec: EnvSpec,
                 state_des: np.ndarray,
                 rew_fcn: RewFcn,
                 success_fcn: callable = None):
        """
        Constructor

        :param env_spec: environment specification of a simulated or real environment
        :param state_des: desired state a.k.a. goal state
        :param rew_fcn: reward function, an instance of a subclass of RewFcn
        :param success_fcn: function to determine if the task was solved, by default (`None`) this task runs endlessly
        """
        if not isinstance(env_spec, EnvSpec):
            raise pyrado.TypeErr(given=env_spec, expected_type=EnvSpec)
        if not isinstance(state_des, np.ndarray):
            raise pyrado.TypeErr(given=state_des, expected_type=np.ndarray)
        if not isinstance(rew_fcn, RewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=RewFcn)

        self._env_spec = env_spec
        self._state_des = state_des
        self._rew_fcn = rew_fcn
        self._success_fcn = success_fcn if success_fcn is not None else never_succeeded

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def state_des(self) -> np.ndarray:
        return self._state_des

    @state_des.setter
    def state_des(self, state_des: np.ndarray):
        if not isinstance(state_des, np.ndarray):
            raise pyrado.TypeErr(given=state_des, expected_type=np.ndarray)
        if not state_des.shape == self.state_des.shape:
            raise pyrado.ShapeErr(given=state_des, expected_match=self.state_des)
        self._state_des = state_des

    @property
    def rew_fcn(self) -> RewFcn:
        return self._rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: RewFcn):
        if not isinstance(rew_fcn, RewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=RewFcn)
        self._rew_fcn = rew_fcn

    def reset(self, env_spec: EnvSpec, state_des: np.ndarray = None, **kwargs):
        """
        Reset the task.

        :param env_spec: environment specification
        :param state_des: new desired state a.k.a. goal state
        :param kwargs: keyword arguments forwarded to the reward function, e.g. the initial state
        """
        # Update the environment specification at every reset of the environment since the spaces could change
        self._env_spec = env_spec

        if state_des is not None:
            self._state_des = state_des

        # Some reward functions scale with the state and action bounds
        self._rew_fcn.reset(state_space=env_spec.state_space, act_space=env_spec.act_space, **kwargs)

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # Operate on state and action errors
        err_state = self._state_des - state
        return self._rew_fcn(err_state, -act, remaining_steps)  # act_des = 0

    def has_succeeded(self, state: np.ndarray) -> bool:
        return self._success_fcn(self.state_des - state)


class RadiallySymmDesStateTask(DesStateTask):
    """
    Task class for moving to a desired state. Operates on the error in state and action.
    In contrast to DesStateTask, a subset of the state is radially symmetric, e.g. and angular position.
    """

    def __init__(self,
                 env_spec: EnvSpec,
                 state_des: np.ndarray,
                 rew_fcn: RewFcn,
                 idcs: Sequence[int],
                 modulation: [float, np.ndarray] = 2*np.pi):
        """
        Constructor

        :param env_spec: environment specification of a simulated or real environment
        :param state_des: desired state a.k.a. goal state
        :param rew_fcn: reward function, an instance of a subclass of RewFcn
        :param idcs: indices of the state dimension(s) to apply the modulation
        :param modulation: factor for the modulo operation, can be specified separately for every `idcs`
        """
        super().__init__(env_spec, state_des, rew_fcn)

        self.idcs = idcs
        self.mod = modulation*np.ones(len(idcs))

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # Modulate the state error
        err_state = self.state_des - state
        err_state[self.idcs] = np.fmod(err_state[self.idcs], self.mod)  # by default map to [-2pi, 2pi]

        # Look at the shortest path to the desired state i.e. desired angle
        err_state[err_state > np.pi] = 2*np.pi - err_state[err_state > np.pi]  # e.g. 360 - (210) = 150
        err_state[err_state < -np.pi] = -2*np.pi - err_state[err_state < -np.pi]  # e.g. -360 - (-210) = -150

        return self.rew_fcn(err_state, -act)  # act_des = 0
