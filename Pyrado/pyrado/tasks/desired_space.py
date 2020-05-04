import numpy as np

import pyrado
from pyrado.spaces.base import Space
from pyrado.utils.data_types import EnvSpec
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import RewFcn, ZeroPerStepRewFcn, MinusOnePerStepRewFcn, PlusOnePerStepRewFcn


class DesSpaceTask(Task):
    """
    Task class for moving to a (fixed) desired state space.
    This task is designed with the idea in mind that it is only important if the state is in the desired (sub)space.
    If the state is in the desired space, the `done` flag is raised. Until then, the step reward is returned.
    """

    def __init__(self,
                 env_spec: EnvSpec,
                 space_des: Space,
                 rew_fcn: RewFcn = ZeroPerStepRewFcn):
        """
        Constructor

        :param env_spec: environment specification of a simulated or real environment
        :param space_des: desired state a.k.a. goal state
        :param rew_fcn: reward function, an instance of a subclass of RewFcn
        """
        if not isinstance(env_spec, EnvSpec):
            raise pyrado.TypeErr(given=env_spec, expected_type=EnvSpec)
        if not isinstance(space_des, Space):
            raise pyrado.TypeErr(given=space_des, expected_type=Space)
        if not isinstance(rew_fcn, (ZeroPerStepRewFcn, PlusOnePerStepRewFcn, MinusOnePerStepRewFcn)):
            raise pyrado.TypeErr(
                given=rew_fcn, expected_type=[ZeroPerStepRewFcn, PlusOnePerStepRewFcn, MinusOnePerStepRewFcn]
            )

        self._env_spec = env_spec
        self._space_des = space_des
        self._rew_fcn = rew_fcn

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def space_des(self) -> Space:
        return self._space_des

    @space_des.setter
    def space_des(self, space_des: Space):
        if not isinstance(space_des, Space):
            raise pyrado.TypeErr(given=space_des, expected_type=Space)
        self._space_des = space_des

    @property
    def rew_fcn(self) -> RewFcn:
        return self._rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: RewFcn):
        if not isinstance(rew_fcn, (ZeroPerStepRewFcn, MinusOnePerStepRewFcn)):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=[ZeroPerStepRewFcn, MinusOnePerStepRewFcn])
        self._rew_fcn = rew_fcn

    def reset(self, env_spec: EnvSpec, space_des: Space = None, **kwargs):
        """
        Reset the task.

        :param env_spec: environment specification
        :param space_des: new desired state a.k.a. goal state
        :param kwargs: keyword arguments forwarded to the reward function, e.g. the initial state
        """
        # Update the environment specification at every reset of the environment since the spaces could change
        self._env_spec = env_spec

        if space_des is not None:
            self._space_des = space_des

        # Some reward functions scale with the state and action bounds
        self._rew_fcn.reset(state_space=env_spec.state_space, act_space=env_spec.act_space, **kwargs)

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # Operate on state and actions
        return self._rew_fcn(state, act, remaining_steps)

    def has_succeeded(self, state: np.ndarray) -> bool:
        return self._space_des.contains(state)
