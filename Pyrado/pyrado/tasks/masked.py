import numpy as np

from pyrado.utils.data_types import EnvSpec
from pyrado.spaces.empty import EmptySpace
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import RewFcn


class MaskedTask(Task):
    """ Task using only a subset of state and actions """

    def __init__(self,
                 env_spec: EnvSpec,
                 wrapped: Task,
                 state_idcs: [str, int],
                 action_idcs: [str, int] = None):
        """
        Constructor

        :param env_spec: environment specification
        :param wrapped: task for the selected part of the state-action space
        :param state_idcs: indices of the selected states
        :param action_idcs: indices of the selected actions
        """
        self._env_spec = env_spec
        self._wrapped = wrapped
        self._state_idcs = state_idcs
        self._action_idcs = action_idcs

        # Written by reset
        self._state_mask = None
        self._action_mask = None

        self.reset(env_spec)

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def state_des(self) -> np.ndarray:
        # The desired state is NaN for masked entries.
        full = np.full(self.env_spec.state_space.shape, np.nan)
        full[self._state_mask] = self._wrapped.state_des
        return full

    @state_des.setter
    def state_des(self, state_des: np.ndarray):
        self._wrapped.state_des = state_des[self._state_mask]

    @property
    def rew_fcn(self) -> RewFcn:
        return self._wrapped.rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: RewFcn):
        self._wrapped.rew_fcn = rew_fcn

    def reset(self, env_spec: EnvSpec, **kwargs):
        self._env_spec = env_spec

        # Determine the masks
        if self._state_idcs is not None:
            self._state_mask = env_spec.state_space.create_mask(self._state_idcs)
        else:
            self._state_mask = np.ones(env_spec.state_space.shape, dtype=np.bool_)
        if self._action_idcs is not None:
            self._action_mask = env_spec.act_space.create_mask(self._action_idcs)
        else:
            self._action_mask = np.ones(env_spec.act_space.shape, dtype=np.bool_)

        # Pass masked state and masked action
        self._wrapped.reset(
            env_spec=EnvSpec(
                env_spec.obs_space,
                env_spec.act_space.subspace(self._action_mask),
                env_spec.state_space.subspace(self._state_mask) if env_spec.state_space is not EmptySpace else EmptySpace),
            **kwargs
        )

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        # Pass masked state and masked action
        return self._wrapped.step_rew(state[self._state_mask], act[self._action_mask], remaining_steps)

    def final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        # Pass masked state and masked action
        return self._wrapped.final_rew(state[self._state_mask], remaining_steps)

    def has_succeeded(self, state: np.ndarray) -> bool:
        # Pass masked state and masked action
        return self._wrapped.has_succeeded(state[self._state_mask])

    def has_failed(self, state: np.ndarray) -> bool:
        # Pass masked state and masked action
        return self._wrapped.has_failed(state[self._state_mask])

    def is_done(self, state: np.ndarray) -> bool:
        # Pass masked state and masked action
        return self._wrapped.is_done(state[self._state_mask])
