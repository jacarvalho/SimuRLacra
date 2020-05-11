from typing import Any

import numpy as np

import pyrado
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, RewFcn
from pyrado.utils.data_types import EnvSpec


class WAMBallInCupTask(Task):
    def __init__(self, env_spec: EnvSpec, sim):
        self._env_spec = env_spec
        self._sim = sim
        self._rew_fcn = ExpQuadrErrRewFcn(Q=np.eye(3), R=np.eye(6))

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def rew_fcn(self) -> RewFcn:
        return self._rew_fcn

    def reset(self, **kwargs: Any):
        pass

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        # state and act are meaningless for reward
        ball_goal_pos = self._sim.data.body_xpos[10].copy()
        ball_pos = self._sim.data.body_xpos[40].copy()
        err_state = ball_goal_pos - ball_pos
        return self._rew_fcn(err_state, np.zeros_like(act))

    def has_succeeded(self, state: np.ndarray) -> bool:
        return False