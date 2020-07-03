import numpy as np

import pyrado
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import StateBasedRewFcn, RewFcn
from pyrado.utils.data_types import EnvSpec


class GoallessTask(Task):
    """ Task which has no desired state or desired space, this runs endlessly """

    def __init__(self, env_spec: EnvSpec, rew_fcn: RewFcn):
        """
        Constructor

        :param env_spec: environment specification
        :param rew_fcn: reward function, an instance of a subclass of RewFcn
        """
        self._env_spec = env_spec
        self._rew_fcn = rew_fcn

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

    def reset(self, env_spec: EnvSpec, **kwargs):
        """
        Reset the task.

        :param env_spec: environment specification
        :param kwargs: keyword arguments forwarded to the reward function, e.g. the initial state
        """
        # Update the environment specification at every reset of the environment since the spaces could change
        self._env_spec = env_spec

        # Some reward functions scale with the state and action bounds
        self._rew_fcn.reset(state_space=env_spec.state_space, act_space=env_spec.act_space, **kwargs)

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # Operate on state and actions
        return self.rew_fcn(state, act, remaining_steps)

    def has_succeeded(self, state: np.ndarray) -> bool:
        return False  # never succeed


class OptimProxyTask(Task):
    """ Task for wrapping classical optimization problems a.k.a. (nonlinear) programming into Pyrado """

    def __init__(self, env_spec: EnvSpec, rew_fcn: StateBasedRewFcn):
        """
        Constructor

        :param env_spec: environment specification
        :param rew_fcn: state-based reward function that maps the state to an scalar value
        """
        assert isinstance(rew_fcn, StateBasedRewFcn)

        self._env_spec = env_spec
        self._rew_fcn = rew_fcn

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def rew_fcn(self) -> StateBasedRewFcn:
        return self._rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: StateBasedRewFcn):
        if not isinstance(rew_fcn, StateBasedRewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=StateBasedRewFcn)
        self._rew_fcn = rew_fcn

    def reset(self, env_spec, **kwargs):
        # Nothing to do
        pass

    def step_rew(self, state: np.ndarray, act: np.ndarray = None, remaining_steps: int = None) -> float:
        # No dependency on the action or the time step here
        return self.rew_fcn(state)

    def has_succeeded(self, state: np.ndarray) -> bool:
        return False  # never succeed
