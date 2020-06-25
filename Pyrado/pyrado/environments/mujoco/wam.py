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
from pyrado.tasks.condition_only import ConditionOnlyTask
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.final_reward import BestStateFinalRewTask, FinalRewTask, FinalRewMode
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn, ExpQuadrErrRewFcn, QuadrErrRewFcn
from pyrado.tasks.sequential import SequentialTasks
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt


class WAMSim(MujocoSimEnv, Serializable):
    """
    WAM Arm from Barrett technologies.

    .. note::
        When using the `reset()` function, always pass a meaningful `init_state`

    .. seealso::
        https://github.com/jhu-lcsr/barrett_model
        http://www.mujoco.org/book/XMLreference.html (e.g. for joint damping)
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
        model_path = osp.join(pyrado.MUJOCO_ASSETS_DIR, 'wam_7dof_base.xml')
        super().__init__(model_path, frame_skip, max_steps, task_args)

        self.camera_config = dict(
            trackbodyid=0,  # id of the body to track
            elevation=-30,  # camera rotation around the axis in the plane
            azimuth=-90  # camera rotation around the camera's vertical axis
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

    def _create_task(self, task_args: dict = None) -> Task:
        state_des = np.concatenate([self.init_qpos.copy(), self.init_qvel.copy()])
        return DesStateTask(self.spec, state_des, ZeroPerStepRewFcn())

    def _mujoco_step(self, act: np.ndarray) -> dict:
        self.sim.data.qfrc_applied[:] = act
        self.sim.step()

        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        self.state = np.concatenate([qpos, qvel])
        return dict()


class WAMBallInCupSim(MujocoSimEnv, Serializable):
    """
    WAM Arm from Barrett technologies for the Ball-in-a-cup task.

    .. note::
        When using the `reset()` function, always pass a meaningful `init_state`

    .. seealso::
        [1] https://github.com/psclklnk/self-paced-rl/tree/master/sprl/envs/ball_in_a_cup.py
    """

    name: str = 'wam-bic'

    def __init__(self,
                 frame_skip: int = 4,
                 max_steps: int = pyrado.inf,
                 stop_on_collision: bool = True,
                 fixed_initial_state: bool = True,
                 task_args: [dict, None] = None):
        """
        Constructor

        :param max_steps: max number of simulation time steps
        :param max_steps: max number of simulation time steps
        :param stop_on_collision: set the `failed` flag in the `dict` returned by `_mujoco_step()` to true, if the ball
                                  collides with something else than the desired parts of the cup. This causes the
                                  episode to end. Keep in mind that in case of a negative step reward and no final
                                  cost on failing, this might result in undesired behavior.
        :param fixed_initial_state: enables/disables deterministic, fixed initial state
        :param task_args: arguments for the task construction
        """
        Serializable._init(self, locals())

        self.fixed_initial_state = fixed_initial_state

        model_path = osp.join(pyrado.MUJOCO_ASSETS_DIR, 'wam_7dof_bic.xml')
        super().__init__(model_path, frame_skip, max_steps, task_args)

        # Desired joint position for the initial state
        self.init_pose_des = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])

        # Controller gains
        self.p_gains = np.array([200.0, 300.0, 100.0, 100.0, 10.0, 10.0, 2.5])
        self.d_gains = np.array([7.0, 15.0, 5.0, 2.5, 0.3, 0.3, 0.05])

        # We access a private attribute since a method like 'model.geom_names[geom_id]' cannot be used because
        # not every geom has a name
        self._collision_geom_ids = [self.model._geom_name2id[name] for name in ['cup_geom1', 'cup_geom2']]
        self._collision_bodies = ['wam/wrist_pitch_link', 'wam/wrist_yaw_link', 'wam/forearm_link',
                                  'wam/upper_arm_link', 'wam/shoulder_pitch_link', 'wam/shoulder_yaw_link',
                                  'wam/base_link']
        self.stop_on_collision = stop_on_collision

        self.camera_config = dict(
            trackbodyid=0,  # id of the body to track
            elevation=-30,  # camera rotation around the axis in the plane
            azimuth=-90  # camera rotation around the camera's vertical axis
        )

    @property
    def torque_space(self) -> Space:
        return self._torque_space

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            cup_scale=1.,  # scaling factor for the radius of the cup [-] (should be >0.65)
            rope_length=0.3,  # length of the rope [m] (previously 0.3103)
            ball_mass=0.021,  # mass of the ball [kg]
            joint_damping=0.05,  # damping of motor joints [N/s] (default value is small)
            joint_stiction=0.1,  # dry friction coefficient of motor joints (reasonable values are 0.1 to 0.6)
        )

    def _create_spaces(self):
        # Initial state space
        # Set the actual stable initial position. This position would be reached after some time using the internal
        # PD controller to stabilize at self.init_pose_des
        # An initial qpos measured on real Barret WAM:
        #   [-3.6523e-05  6.4910e-01  4.4244e-03  1.4211e+00  8.8864e-03 -2.7763e-01 -1.5309e+00]
        self.init_qpos[:7] = np.array([0., 0.65, 0., 1.41, 0., -0.28, -1.57])
        # Set the angle of the first rope segment relative to the cup bottom plate
        self.init_qpos[7] = -0.21
        # The initial position of the ball in cartesian coordinates
        init_ball_pos = np.array([0.828, 0., 1.131])
        init_state = np.concatenate([self.init_qpos, self.init_qvel, init_ball_pos])
        if self.fixed_initial_state:
            self._init_space = SingularStateSpace(init_state)
        else:
            # Add plus/minus one degree to each motor joint and the first rope segment joint
            init_state_up = init_state.copy()
            init_state_up[:7] += np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*np.pi/180
            init_state_lo = init_state.copy()
            init_state_lo[:7] -= np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*np.pi/180
            self._init_space = BoxSpace(init_state_lo, init_state_up)

        # State space
        state_shape = init_state.shape
        state_up = np.full(state_shape, pyrado.inf)
        state_lo = np.full(state_shape, -pyrado.inf)
        # Ensure that joint limits of the arm are not reached (up to 5 degree)
        state_up[:7] = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7]) - 5*np.pi/180
        state_lo[:7] = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7]) + 5*np.pi/180
        self._state_space = BoxSpace(state_lo, state_up)

        # Torque space
        max_torque = np.array([150.0, 125.0, 40.0, 60.0, 5.0, 5.0, 2.0])
        self._torque_space = BoxSpace(-max_torque, max_torque)

        # Action space (PD controller on 3 joint positions and velocities)
        act_up = np.array([1.985, np.pi, np.pi/2, 10*np.pi, 10*np.pi, 10*np.pi])
        act_lo = np.array([-1.985, -0.9, -np.pi/2, -10*np.pi, -10*np.pi, -10*np.pi])
        self._act_space = BoxSpace(act_lo, act_up,  # [rad, rad, rad, rad/s, rad/s, rad/s]
                                   labels=[r'$q_{1,des}$', r'$q_{3,des}$', r'$q_{5,des}$',
                                           r'$\dot{q}_{1,des}$', r'$\dot{q}_{3,des}$', r'$\dot{q}_{5,des}$'])

        # Observation space (normalized time)
        self._obs_space = BoxSpace(np.array([0.]), np.array([1.]), labels=['$t$'])

    def _create_task(self, task_args: dict) -> Task:
        return ParallelTasks([self._create_main_task(task_args), self._create_deviation_task(task_args)])

    def _create_main_task(self, task_args: dict) -> Task:
        # Create a DesStateTask that masks everything but the ball position
        idcs = list(range(self.state_space.flat_dim - 3, self.state_space.flat_dim))  # Cartesian ball position
        spec = EnvSpec(
            self.spec.obs_space,
            self.spec.act_space,
            self.spec.state_space.subspace(self.spec.state_space.create_mask(idcs))
        )

        # If we do not use copy(), state_des coming from MuJoCo is a reference and updates automatically at each step.
        # Note: sim.forward() + get_body_xpos() results in wrong output for state_des, as sim has not been updated to
        # init_space.sample(), which is first called in reset()

        if task_args.get('sparse_rew_fcn', False):
            # Binary final reward task
            main_task = FinalRewTask(
                ConditionOnlyTask(spec, condition_fcn=self.check_ball_in_cup, is_success_condition=True),
                mode=FinalRewMode(always_positive=True), factor=1
            )
            # Yield -1 on fail after the main task ist done (successfully or not)
            dont_fail_after_succ_task = FinalRewTask(
                GoallessTask(spec, ZeroPerStepRewFcn()),
                mode=FinalRewMode(always_negative=True), factor=1
            )

            # Augment the binary task with an endless dummy task, to avoid early stopping of the
            task = SequentialTasks((main_task, dont_fail_after_succ_task))

            return MaskedTask(self.spec, task, idcs)

        else:
            # If we do not use copy(), state_des is a reference to passed body and updates automatically at each step
            state_des = self.sim.data.get_site_xpos('cup_goal')  # this is a reference
            rew_fcn = ExpQuadrErrRewFcn(
                Q=task_args.get('Q', np.diag([2e1, 1e-2, 2e1])),  # distance ball - cup; shouldn't move in y-direction
                R=task_args.get('R', np.zeros((spec.act_space.flat_dim, spec.act_space.flat_dim)))
            )
            task = DesStateTask(spec, state_des, rew_fcn)

            # Wrap the masked DesStateTask to add a bonus for the best state in the rollout
            return BestStateFinalRewTask(
                MaskedTask(self.spec, task, idcs),
                max_steps=self.max_steps, factor=task_args.get('final_factor', 1.)
            )

    def _create_deviation_task(self, task_args: dict) -> Task:
        # Create a DesStateTask that masks everything but the actuated joint angles
        idcs = [1, 3, 5]  # see act in _mujoco_step()
        spec = EnvSpec(
            self.spec.obs_space,
            self.spec.act_space,
            self.spec.state_space.subspace(self.spec.state_space.create_mask(idcs))
        )

        state_des = self.sim.data.qpos[1:7:2].copy()  # actual init pose of the controlled joints
        rew_fcn = QuadrErrRewFcn(
            Q=np.diag([8e-2, 5e-2, 5e-2]),
            R=np.zeros((spec.act_space.flat_dim, spec.act_space.flat_dim))
        )
        task = DesStateTask(spec, state_des, rew_fcn)

        return MaskedTask(self.spec, task, idcs)

    def _adapt_model_file(self, xml_model: str, domain_param: dict) -> str:
        # First replace special domain parameters
        cup_scale = domain_param.pop('cup_scale', None)
        rope_length = domain_param.pop('rope_length', None)
        joint_stiction = domain_param.pop('joint_stiction', None)

        if cup_scale is not None:
            # See [1, l.93-96]
            xml_model = xml_model.replace('[scale_mesh]', str(cup_scale*0.001))
            xml_model = xml_model.replace('[pos_mesh]', str(0.055 - (cup_scale - 1.)*0.023))
            xml_model = xml_model.replace('[pos_goal]', str(0.1165 + (cup_scale - 1.)*0.0385))
            xml_model = xml_model.replace('[size_cup]', str(cup_scale*0.038))
            xml_model = xml_model.replace('[size_cup_inner]', str(cup_scale*0.03))

        if rope_length is not None:
            # The rope consists of 30 capsules
            xml_model = xml_model.replace('[pos_capsule]', str(rope_length/30))
            # Each joint is at the top of each capsule (therefore negative direction from center)
            xml_model = xml_model.replace('[pos_capsule_joint]', str(-rope_length/60))
            # Pure visualization component
            xml_model = xml_model.replace('[size_capsule_geom]', str(rope_length/72))

        if rope_length is not None:
            # Amplify joint stiction (dry friction) for the stronger motor joints
            xml_model = xml_model.replace('[stiction_1]', str(4*joint_stiction))
            xml_model = xml_model.replace('[stiction_3]', str(2*joint_stiction))
            xml_model = xml_model.replace('[stiction_5]', str(joint_stiction))
        # Resolve mesh directory and replace the remaining domain parameters
        return super()._adapt_model_file(xml_model, domain_param)

    def _mujoco_step(self, act: np.ndarray) -> dict:
        # Get the desired positions and velocities for the selected joints
        qpos_des = self.init_pose_des.copy()  # the desired trajectory is relative to self.init_pose_des
        np.add.at(qpos_des, [1, 3, 5], act[:3])
        qvel_des = np.zeros_like(qpos_des)
        np.add.at(qvel_des, [1, 3, 5], act[3:])

        # Compute the position and velocity errors
        err_pos = qpos_des - self.state[:7]
        err_vel = qvel_des - self.state[self.model.nq:self.model.nq + 7]

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

        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        ball_pos = self.sim.data.get_body_xpos('ball').copy()
        self.state = np.concatenate([qpos, qvel, ball_pos])

        # If desired, check for collisions of the ball with the robot
        ball_collided = self.check_ball_collisions() if self.stop_on_collision else False

        return dict(
            qpos_des=qpos_des, qvel_des=qvel_des, qpos=qpos[:7], qvel=qvel[:7], ball_pos=ball_pos,
            cup_pos=self.sim.data.get_site_xpos('cup_goal').copy(), failed=mjsim_crashed or ball_collided
        )

    def check_ball_collisions(self, verbose: bool = False) -> bool:
        """
        Check if an undesired collision with the ball occurs.

        :param verbose: print messages on collision
        :return: `True` if the ball collides with something else than the central parts of the cup
        """
        for i in range(self.sim.data.ncon):
            # Get current contact object
            contact = self.sim.data.contact[i]

            # Extract body-id and body-name of both contact geoms
            body1 = self.model.geom_bodyid[contact.geom1]
            body1_name = self.model.body_names[body1]
            body2 = self.model.geom_bodyid[contact.geom2]
            body2_name = self.model.body_names[body2]

            # Evaluate if the ball collides with part of the WAM (collision bodies)
            # or the connection of WAM and cup (geom_ids)
            c1 = body1_name == 'ball' and (body2_name in self._collision_bodies or
                                           contact.geom2 in self._collision_geom_ids)
            c2 = body2_name == 'ball' and (body1_name in self._collision_bodies or
                                           contact.geom1 in self._collision_geom_ids)
            if c1 or c2:
                if verbose:
                    print_cbt(f'Undesired collision of {body1_name} and {body2_name} detected!', 'y')
                return True

        return False

    def check_ball_in_cup(self, *args, verbose: bool = False):
        """
        Check if the ball is in the cup.

        :param verbose: print messages when ball is in the cup
        :return: `True` if the ball is in the cup
        """
        for i in range(self.sim.data.ncon):
            # Get current contact object
            contact = self.sim.data.contact[i]

            # Extract body-id and body-name of both contact geoms
            body1 = self.model.geom_bodyid[contact.geom1]
            body1_name = self.model.body_names[body1]
            body2 = self.model.geom_bodyid[contact.geom2]
            body2_name = self.model.body_names[body2]

            # Evaluate if the ball collides with part of the WAM (collision bodies)
            # or the connection of WAM and cup (geom_ids)
            cup_inner_id = self.model._geom_name2id['cup_inner']
            c1 = body1_name == 'ball' and contact.geom2 == cup_inner_id
            c2 = body2_name == 'ball' and contact.geom1 == cup_inner_id
            if c1 or c2:
                if verbose:
                    print_cbt(f'The ball is in the cup at time step {self.curr_step}.', 'y')
                return True

        return False

    def observe(self, state: np.ndarray) -> np.ndarray:
        # Only observe the normalized time
        return np.array([self._curr_step/self.max_steps])
