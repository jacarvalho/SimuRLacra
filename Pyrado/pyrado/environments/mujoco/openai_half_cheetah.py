import os.path as osp
import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.mujoco.base import MujocoSimEnv
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.reward_functions import ForwardVelocityRewFcn


class HalfCheetahSim(MujocoSimEnv, Serializable):
    """
    The Half-Cheetah MuJoCo simulation environment where a planar simplified cheetah-like robot tries to run forward.

    .. seealso::
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py
    """

    name: str = 'cth'

    def __init__(self, frame_skip: int = 5, max_steps: int = 1000, task_args: [dict, None] = None):
        """
        Constructor

        :param frame_skip: number of frames for holding the same action, i.e. multiplier of the time step size
        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction, e.g `dict(fwd_rew_weight=1.)`
        """
        # Call MujocoSimEnv's constructor
        model_path = osp.join(osp.dirname(__file__), 'assets', 'openai_half_cheetah.xml')
        super().__init__(model_path, frame_skip, max_steps, task_args)

        self.camera_config = dict(distance=5.0)

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        """
        Get the nominal a.k.a. default domain parameters.

        .. seealso::
            http://www.mujoco.org/book/XMLreference.html#geom
            http://www.mujoco.org/book/computation.html#coContact
        """
        return dict(
            total_mass=14,
            tangential_friction_coeff=0.4,
            torsional_friction_coeff=0.1,
            rolling_friction_coeff=0.1,
            reset_noise_halfspan=0.1
        )

    def _create_spaces(self):
        # Action
        act_bounds = self.model.actuator_ctrlrange.copy().T
        self._act_space = BoxSpace(*act_bounds, labels=['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot'])

        # State
        state_shape = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).shape
        max_state = np.full(state_shape, pyrado.inf)
        self._state_space = BoxSpace(-max_state, max_state)

        # Initial state
        noise_halfspan = self.domain_param['reset_noise_halfspan']
        min_init_qpos = self.init_qpos - np.full_like(self.init_qpos, noise_halfspan)
        max_init_qpos = self.init_qpos + np.full_like(self.init_qpos, noise_halfspan)
        min_init_qvel = self.init_qvel - np.full_like(self.init_qpos, noise_halfspan)
        max_init_qvel = self.init_qvel + np.full_like(self.init_qpos, noise_halfspan)
        min_init_state = np.concatenate([min_init_qpos, min_init_qvel]).ravel()
        max_init_state = np.concatenate([max_init_qpos, max_init_qvel]).ravel()
        self._init_space = BoxSpace(min_init_state, max_init_state)

        # Observation
        obs_shape = self.observe(max_state).shape
        max_obs = np.full(obs_shape, pyrado.inf)
        self._obs_space = BoxSpace(-max_obs, max_obs)

    def _create_task(self, task_args: [dict, None] = None) -> Task:
        if task_args is None:
            task_args = dict(fwd_rew_weight=1., ctrl_cost_weight=0.1)
        return GoallessTask(self.spec, ForwardVelocityRewFcn(self._dt, idx_fwd=0, **task_args))

    def _mujoco_step(self, act: np.ndarray) -> dict:
        self.sim.data.ctrl[:] = act
        # Changelog: frame_skip is now directly passed to self.sim
        self.sim.step()

        pos = self.sim.data.qpos.copy()
        vel = self.sim.data.qvel.copy()
        self.state = np.concatenate([pos, vel])

        return dict()

    def observe(self, state: np.ndarray) -> np.ndarray:
        # Ignore horizontal position to maintain translational invariance
        return state[1:].copy()
