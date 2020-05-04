import mujoco_py
import numpy as np
import os.path as osp
from init_args_serializer import Serializable

import pyrado
from pyrado.spaces.base import Space
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.base import Task
from pyrado.environments.mujoco.base import MujocoSimEnv
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn


class WAMSim(MujocoSimEnv, Serializable):
    # TODO: this class is work in progress...
    """
    WAM Arm from Barrett technologies.

    .. note::
        If using `reset()` function, always pass a meaningful `init_state`

    .. seealso::
        https://github.com/jhu-lcsr/barrett_model
    """

    name: str = 'wam'

    def __init__(self,
                 frame_skip: int = 1,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None):
        """
        Constructor

        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction
        """
        model_path = osp.join(pyrado.MUJOCO_ASSETS_DIR, 'wam_7dof.xml')
        super().__init__(model_path, frame_skip, max_steps, task_args)

        self.camera_config = dict(
            trackbodyid=0,  # id of the body to track
            elevation=-30,  # camera rotation around the axis in the plane
            azimuth=180  # camera rotation around the camera's vertical axis
        )

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict()

    def _create_spaces(self):
        # Action space
        max_act = np.array([150., 125., 40., 60., 5., 5., 2.])
        self._act_space = BoxSpace(-max_act, max_act)

        # State space
        state_shape = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).shape
        max_state = np.full(state_shape, pyrado.inf)
        self._state_space = BoxSpace(-max_state, max_state)

        # Initial state space
        self._init_space = self._state_space.copy()

        # Observation space
        obs_shape = self.observe(max_state).shape
        max_obs = np.full(obs_shape, pyrado.inf)
        self._obs_space = BoxSpace(-max_obs, max_obs)

    def _create_task(self, task_args: [dict, None] = None) -> Task:
        # TODO: Formulate proper task/reward
        state_des = np.concatenate([self.init_qpos.copy(), self.init_qvel.copy()])
        return DesStateTask(self.spec, state_des, ZeroPerStepRewFcn())

    def _mujoco_step(self, act: np.ndarray):
        self.sim.data.qfrc_applied[:] = act
        self.sim.step()

        pos = self.sim.data.qpos.copy()
        vel = self.sim.data.qvel.copy()
        self.state = np.concatenate([pos, vel])


class WAMBallInCupSim(MujocoSimEnv, Serializable):
    # TODO: this class is work in progress...
    """
    WAM Arm from Barrett technologies for the Ball-in-a-cup task.

    .. note::
        If using `reset()` function, always pass a meaningful `init_state`

    .. seealso::
        https://github.com/psclklnk/self-paced-rl/tree/master/sprl/envs
    """

    name: str = 'wam-bic'

    def __init__(self,
                 frame_skip: int = 4,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None):
        """
        Constructor

        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction
        """
        model_path = osp.join(pyrado.MUJOCO_ASSETS_DIR, 'wam_cup.xml')
        super().__init__(model_path, frame_skip, max_steps, task_args)

        self.p_gains = np.array([200, 300, 100, 100, 10, 10, 2.5])
        self.d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

        self.camera_config = dict(
            trackbodyid=0,  # id of the body to track
            elevation=-30,  # camera rotation around the axis in the plane
            azimuth=180  # camera rotation around the camera's vertical axis
        )

    @property
    def torque_space(self) -> Space:
        return self._torque_space

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict()

    def _create_spaces(self):
        # Torque space
        max_torque = np.array([150., 125., 40., 60., 5., 5., 2.])
        self._torque_space = BoxSpace(-max_torque, max_torque)

        # State space
        state_shape = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).shape
        max_state = np.full(state_shape, pyrado.inf)
        self._state_space = BoxSpace(-max_state, max_state)

        # Initial state space
        init_pos = self.init_qpos.copy()
        init_pos[:7] = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])  # arm position
        init_state = np.concatenate([init_pos, self.init_qvel.copy()]).ravel()
        self._init_space = SingularStateSpace(init_state)

        # Action space
        max_act = np.full((6,), pyrado.inf)
        self._act_space = BoxSpace(-max_act, max_act)

        # Observation space
        self._obs_space = BoxSpace(np.array([0]), np.array([self.max_steps]))

    def _create_task(self, task_args: [dict, None] = None) -> Task:
        # TODO: Formulate proper task/reward
        # TODO: Desired position (slightly above cup_base) - velocity should be perpendicular to the surface rim of the cup
        # check if velocity is downward 
        ball_pos = self.sim.data.body_xpos[40].copy()
        state_des = np.concatenate([self.init_qpos.copy(), self.init_qvel.copy()])
        return DesStateTask(self.spec, state_des, ZeroPerStepRewFcn())

    def _mujoco_step(self, act: np.ndarray):
        # act = (des_qpos, des_qvel) with dim=(6,) is zero centered, therefore the init_pos is added
        des_pos = self._init_space.sample_uniform()[:7]
        des_pos[1] += act[0]
        des_pos[3] += act[1]
        des_pos[5] += act[2]
        des_vel = np.zeros_like(des_pos)
        des_vel[1] += act[3]
        des_vel[3] += act[4]
        des_vel[5] += act[5]

        # Compute error on position and velocity
        err_pos = des_pos - self.state[:7]
        err_vel = des_vel - self.state[:7]

        # Compute torques
        torque = self.p_gains * err_pos + self.d_gains * err_vel
        torque = self.torque_space.project_to(torque)

        # Apply torque
        self.sim.data.qfrc_applied[:7] = torque
        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException as e:
            print(e)
            self.reset()

        pos = self.sim.data.qpos.copy()
        vel = self.sim.data.qvel.copy()
        self.state = np.concatenate([pos, vel])

    def observe(self, state: np.ndarray) -> np.ndarray:
        return np.array([self._curr_step / self.max_steps])
