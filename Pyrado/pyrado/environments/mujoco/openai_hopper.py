import os.path as osp
import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.mujoco.base import MujocoSimEnv
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.reward_functions import ForwardVelocityRewFcn, CombinedRewFcn, PlusOnePerStepRewFcn


class HopperSim(MujocoSimEnv, Serializable):
    """
    The Hopper MuJoCo simulation environment where a planar simplified one-legged robot tries to run forward.

    .. note::
        In contrast to the OpenAI Gym MoJoCo environments, Pyrado enables the randomization of the hoppers "healthy"
        state range. Moreover, the state space is constrained to the this part of the state space. Note the in the
        original environment, the `terminate_when_unhealthy` is `True` by default.

    .. seealso::
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper_v3.py
    """

    name: str = 'hop'

    def __init__(self, frame_skip: int = 5, max_steps: int = 1000, task_args: [dict, None] = None):
        """
        Constructor

        :param frame_skip: number of frames for holding the same action, i.e. multiplier of the time step size
        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction, e.g `dict(fwd_rew_weight=1.)`
        """
        # Call MujocoSimEnv's constructor
        model_path = osp.join(osp.dirname(__file__), 'assets', 'openai_hopper.xml')
        super().__init__(model_path, frame_skip, max_steps, task_args)

        self.camera_config = dict(
            trackbodyid=2,
            distance=3.,
            lookat=np.array((0.0, 0.0, 1.15)),
            elevation=-20.0
        )

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            state_bound=100.,
            z_lower_bound=0.7,
            angle_bound=0.2,
            foot_friction_coeff=2.,
            reset_noise_halfspan=5e-3
        )

    def _create_spaces(self):
        # Action
        act_bounds = self.model.actuator_ctrlrange.copy().T
        self._act_space = BoxSpace(*act_bounds, labels=['thigh', 'leg', 'foot'])

        # State
        n = self.init_qpos.size + self.init_qvel.size
        min_state = -self.domain_param['state_bound']*np.ones(n)
        min_state[0] = -pyrado.inf  # ignore forward position
        min_state[1] = self.domain_param['z_lower_bound']
        min_state[2] = -self.domain_param['angle_bound']
        max_state = self.domain_param['state_bound']*np.ones(n)
        max_state[0] = +pyrado.inf  # ignore forward position
        max_state[2] = self.domain_param['angle_bound']
        self._state_space = BoxSpace(min_state, max_state)

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
            task_args = dict(fwd_rew_weight=1., ctrl_cost_weight=1e-3)
        rew_fcn = CombinedRewFcn([
            ForwardVelocityRewFcn(self._dt, idx_fwd=0, **task_args),
            PlusOnePerStepRewFcn()  # equivalent to the "healthy_reward"
        ])
        return GoallessTask(self.spec, rew_fcn)

    def _mujoco_step(self, act: np.ndarray):
        self.sim.data.ctrl[:] = act
        # Alternatively: pass `frame_skip` as `nsubsteps` argument in MjSim constructor..
        # ..instead of calling sim.step() multiple times
        for _ in range(self.frame_skip):
            self.sim.step()

        pos = self.sim.data.qpos.copy()
        vel = self.sim.data.qvel.copy()
        self.state = np.concatenate([pos, vel])

        return dict()

    def observe(self, state: np.ndarray) -> np.ndarray:
        # Clip velocity
        pos = state[:self.model.nq]
        vel = np.clip(state[self.model.nq:], -10., 10.)

        # Ignore horizontal position to maintain translational invariance
        return np.concatenate([pos[1:], vel])
