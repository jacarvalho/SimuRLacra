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
from pyrado.tasks.environment_specific import WAMBallInCupTask
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

    def _mujoco_step(self, act: np.ndarray) -> dict:
        self.sim.data.qfrc_applied[:] = act
        self.sim.step()

        pos = self.sim.data.qpos.copy()
        vel = self.sim.data.qvel.copy()
        self.state = np.concatenate([pos, vel])

        return dict()


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

        # Desired position for the initial state
        self.init_des_pos = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])

        # Controller gains
        self.p_gains = np.array([200.0, 300.0, 100.0, 100.0, 10.0, 10.0, 2.5])
        self.d_gains = np.array([7.0, 15.0, 5.0, 2.5, 0.3, 0.3, 0.05])

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
        max_torque = np.array([150.0, 125.0, 40.0, 60.0, 5.0, 5.0, 2.0])
        self._torque_space = BoxSpace(-max_torque, max_torque)

        # State space
        state_shape = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).shape
        max_state = np.full(state_shape, pyrado.inf)
        self._state_space = BoxSpace(-max_state, max_state)

        # Initial state space
        # Set the actual stable initial position. This position would be reached after som time using the internal
        # PD controller to stabilize at self.init_des_pos
        np.put(self.init_qpos, [1, 3, 5, 6, 7], [0.6519, 1.409, -0.2827, -1.57, -0.2115])
        init_state = np.concatenate([self.init_qpos.copy(), self.init_qvel.copy()]).ravel()
        self._init_space = SingularStateSpace(init_state)

        # Action space (PD controller on 3 joint positions and velocities)
        max_act = np.array([np.pi, np.pi, np.pi,  # [rad, rad, rad, ...
                            10*np.pi, 10*np.pi, 10*np.pi])  # ... rad/s, rad/s, rad/s]
        self._act_space = BoxSpace(-max_act, max_act)

        # Observation space (normalized time)
        self._obs_space = BoxSpace(np.array([0.]), np.array([1.]))

    def _create_task(self, task_args: [dict, None] = None) -> Task:
        return WAMBallInCupTask(self.spec, self.sim)

    def _mujoco_step(self, act: np.ndarray) -> dict:
        # Get the desired positions and velocities for the selected joints
        des_pos = self.init_des_pos.copy()  # the desired trajectory is relative to self.init_des_pos
        np.add.at(des_pos, [1, 3, 5], act[:3])
        des_vel = np.zeros_like(des_pos)
        np.add.at(des_vel, [1, 3, 5], act[3:])

        # Compute the position and velocity errors
        err_pos = des_pos - self.state[:7]
        err_vel = des_vel - self.state[self.model.nq:self.model.nq+7]

        # Compute the torques (PD controller)
        torque = self.p_gains * err_pos + self.d_gains * err_vel
        torque = self.torque_space.project_to(torque)

        # Apply the torques to the robot
        self.sim.data.qfrc_applied[:7] = torque
        try:
            self.sim.step()
            mjsim_crashed = False
        except mujoco_py.builder.MujocoException:
            # When MuJoCo recognized instabilities in the simulation, it simply kills it.
            # Instead, we want the episode to end with a failure
            mjsim_crashed = True
            # self.reset()

        pos = self.sim.data.qpos.copy()
        vel = self.sim.data.qvel.copy()
        self.state = np.concatenate([pos, vel])

        return dict(des_pos=des_pos, des_vel=des_vel, pos=pos[:7], vel=vel[:7], failed=mjsim_crashed)

    def observe(self, state: np.ndarray) -> np.ndarray:
        return np.array([self._curr_step / self.max_steps])
