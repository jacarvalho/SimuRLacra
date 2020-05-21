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
from pyrado.tasks.final_reward import BestStateFinalRewTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn, ExpQuadrErrRewFcn
from pyrado.utils.data_types import EnvSpec


class WAMSim(MujocoSimEnv, Serializable):
    """
    WAM Arm from Barrett technologies.

    .. note::
        When using the `reset()` function, always pass a meaningful `init_state`

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

        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        self.state = np.concatenate([qpos, qvel])
        return dict()


class WAMBallInCupSim(MujocoSimEnv, Serializable):
    """
    WAM Arm from Barrett technologies for the Ball-in-a-cup task.

    .. note::
        When using the `reset()` function, always pass a meaningful `init_state`

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

        # Desired joint position for the initial state
        self.init_pose_des = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])

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
        return dict(
            cup_scale=1.,  # scaling factor for the radius of the cup
            rope_length=0.3103,  # length of the rope
            ball_mass=0.021  # mass of the ball
        )

    def _create_spaces(self):
        # Torque space
        max_torque = np.array([150.0, 125.0, 40.0, 60.0, 5.0, 5.0, 2.0])
        self._torque_space = BoxSpace(-max_torque, max_torque)

        # Initial state space
        # Set the actual stable initial position. This position would be reached after some time using the internal ...
        # ... PD controller to stabilize at self.init_pose_des
        np.put(self.init_qpos, [1, 3, 5, 6, 7], [0.6519, 1.409, -0.2827, -1.57, -0.2115])
        init_ball_pos = np.array([0., -0.8566, 0.85391])
        init_state = np.concatenate([self.init_qpos, self.init_qvel, init_ball_pos])
        self._init_space = SingularStateSpace(init_state)

        # State space
        state_shape = init_state.shape
        max_state = np.full(state_shape, pyrado.inf)
        self._state_space = BoxSpace(-max_state, max_state)

        # Action space (PD controller on 3 joint positions and velocities)
        max_act = np.array([np.pi, np.pi, np.pi,  # [rad, rad, rad, ...
                            10*np.pi, 10*np.pi, 10*np.pi])  # ... rad/s, rad/s, rad/s]
        self._act_space = BoxSpace(-max_act, max_act,
                                   labels=[r'$q_{1,des}$', r'$q_{3,des}$', r'$q_{5,des}$',
                                           r'$\dot{q}_{1,des}$', r'$\dot{q}_{3,des}$', r'$\dot{q}_{5,des}$'])

        # Observation space (normalized time)
        self._obs_space = BoxSpace(np.array([0.]), np.array([1.]), labels=['$t$'])

    def _create_task(self, task_args: [dict, None] = None) -> Task:
        # Create a DesStateTask that masks everything but the ball position
        idcs = list(range(self.state_space.flat_dim-3, self.state_space.flat_dim))  # Cartesian ball position of [x, y, z]
        spec = EnvSpec(
            self.spec.obs_space,
            self.spec.act_space,
            self.spec.state_space.subspace(self.spec.state_space.create_mask(idcs))
        )
        # Original idea
        # self.sim.forward()  # need to call forward to get a non-zero body position
        # state_des = self.sim.data.get_body_xpos('B0').copy()
        # But
        # If we do not use copy(), state_des is a reference and updates automatically at each step
        # sim.forward() + get_body_xpos() results in wrong output for state_des, as sim has not been updated to
        # init_space.sample(), which is first called in reset()
        # Now
        state_des = np.array([0., -0.8566, 1.164])
        rew_fcn = ExpQuadrErrRewFcn(Q=10.*np.eye(3), R=1e-2*np.eye(6))
        dst = DesStateTask(spec, state_des, rew_fcn)

        # Wrap the masked DesStateTask to add a bonus for the best state in the rollout
        if task_args is None:
            task_args = dict(factor=1.)
        return BestStateFinalRewTask(
            MaskedTask(self.spec, dst, idcs),
            max_steps=self.max_steps, factor=task_args.get('factor', 1.)
        )

    def _mujoco_step(self, act: np.ndarray) -> dict:
        # Get the desired positions and velocities for the selected joints
        des_qpos = self.init_pose_des.copy()  # the desired trajectory is relative to self.init_pose_des
        np.add.at(des_qpos, [1, 3, 5], act[:3])
        des_qvel = np.zeros_like(des_qpos)
        np.add.at(des_qvel, [1, 3, 5], act[3:])

        # Compute the position and velocity errors
        err_pos = des_qpos - self.state[:7]
        err_vel = des_qvel - self.state[self.model.nq:self.model.nq + 7]

        # Compute the torques (PD controller)
        torque = self.p_gains*err_pos + self.d_gains*err_vel
        torque = self.torque_space.project_to(torque)

        # Apply the torques to the robot
        self.sim.data.qfrc_applied[:7] = torque
        try:
            self.sim.step()
            mjsim_crashed = False
        except mujoco_py.builder.MujocoException:
            # When MuJoCo recognized instabilities in the simulation, it simply kills it
            # Instead, we want the episode to end with a failure
            mjsim_crashed = True

        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        ball_pos = self.sim.data.get_body_xpos('ball').copy()
        self.state = np.concatenate([qpos, qvel, ball_pos])

        # Update task's desired state
        state_des = np.zeros_like(self.state)  # needs to be of same dimension as self.state since it is masked later
        state_des[-3:] = self.sim.data.get_body_xpos('B0').copy()
        self._task.wrapped_task.state_des = state_des

        return dict(
            des_qpos=des_qpos, des_qvel=des_qvel, qpos=qpos[:7], qvel=qvel[:7], ball_pos=ball_pos,
            state_des=state_des[-3:], failed=mjsim_crashed
        )

    def observe(self, state: np.ndarray) -> np.ndarray:
        # Only observe the normalized time
        return np.array([self._curr_step/self.max_steps])
