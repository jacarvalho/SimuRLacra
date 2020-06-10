import numpy as np
import robcom_python as robcom

import pyrado
from pyrado.environments.base import Env
from pyrado.spaces import BoxSpace
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


class WAMBallInCupReal(Env):
    """
    Class for the real Barrett WAM

    Uses robcom 2.0 and specifically robcom's GoTo process to execute a trajectory given by desired joint positions.
    The process is only executed on the real system after `max_steps` has been reached to avoid possible latency,
    but at the same time mimic the usual step-based environment behavior.
    """

    name: str = 'wam-real'

    def __init__(self,
                 dt: float = 1/500.,
                 max_steps: int = pyrado.inf,
                 ip: str = '192.168.2.2',
                 poses_des: [np.ndarray, None] = None):
        """
        Constructor

        :param dt: sampling time interval
        :param max_steps: maximum number of time steps
        :param ip: IP address of the PC controlling the Barrett WAM
        :param poses_des: desired joint poses as ndarray of shape (., 3)
        """
        # Call the base class constructor to initialize fundamental members
        super().__init__(dt, max_steps)

        # Desired joint position for the initial state
        self.init_pose_des = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])

        # Connect to client
        self._client = robcom.Client()
        self._client.start(ip, 2013)  # IP address and port
        print_cbt('Connected to the Barret WAM client.', 'c', bright=True)
        self._gt = None  # Goto command

        # Initialize spaces
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._create_spaces()

        # Initialize task
        self._task = self._create_task(dict())

        # Desired trajectory
        if not poses_des.shape[1] == 7:
            raise pyrado.ShapeErr(given=poses_des.shape[1], expected_match=self.init_pose_des)
        self.poses_des = poses_des

    @property
    def state_space(self) -> Space:
        return self._state_space

    @property
    def obs_space(self) -> Space:
        return self._obs_space

    @property
    def act_space(self) -> Space:
        return self._act_space

    @property
    def task(self) -> Task:
        return self._task

    def _create_task(self, task_args: dict) -> Task:
        # The wrapped task acts as a dummy and carries the FinalRewTask s
        return FinalRewTask(GoallessTask(self.spec, ZeroPerStepRewFcn()), mode=FinalRewMode.user_input)

    def _create_spaces(self):
        # State space
        state_shape = self.init_pose_des.shape
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

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Create robcom GoTo process
        gt = self._client.create(robcom.Goto, 'RIGHT_ARM', '')

        # Move to initial state within 5 seconds
        gt.add_step(5., self.init_pose_des)
        print_cbt('Moving the Barret WAM to the initial position.', 'c', bright=True)

        # Start process and wait for completion
        gt.start()
        gt.wait_for_completion()
        print_cbt('Reached the initial position.', 'c')

        # Reset time steps
        self._curr_step = 0

        return self.observe(self.state)

    def step(self, act: np.ndarray) -> tuple:
        info = dict(t=self._curr_step*self._dt, act_raw=act)

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)  # always 0 for wam-bic-real

        act = self._limit_act(act)

        if self.poses_des is not None and self._curr_step < self.poses_des.shape[0]:
            # Use given desired trajectory if given and time step does no exceed its length
            des_qpos = self.poses_des[self._curr_step]
        else:
            # Otherwise use the action given by a policy
            des_qpos = self.init_pose_des.copy()  # keep the initial joint angles deselected joints
            np.add.at(des_qpos, [1, 3, 5], act[:3])  # the policy operates on joint 1, 3 and 5

        # Create robcom GoTo process at the first time step TODO @Christian: possible move to the end of reset()?
        if self._curr_step == 0:
            self._gt = self._client.create(robcom.Goto, 'RIGHT_ARM', '')

        # Add desired joint position as step to the process
        self._gt.add_step(self.dt, des_qpos)
        self._curr_step += 1

        # A GoallessTask only signals done when has_failed() is true, i.e. the the state is out of bounds
        done = self._task.is_done(self.state)  # always false for wam-bic-real

        # Only start execution of process when all desired poses have been added to process
        # i.e. `max_steps` has been reached
        if self._curr_step >= self._max_steps:
            done = True
            print_cbt('Executing trajectory on Barret WAM.', 'c')
            self._gt.start()
            self._gt.wait_for_completion()
            print_cbt('Finished execution.', 'c')

        # Add final reward if done
        if done:
            # Ask the user to enter the final reward
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info

    def render(self, mode: RenderMode, render_step: int = 1):
        # Skip all rendering
        pass

    def close(self):
        self._client.close()
        print_cbt('Closed the connection to the Barrett WAM.', 'c', bright=True)

    def observe(self, state: np.ndarray) -> np.ndarray:
        # Only observe the normalized time
        return np.array([self._curr_step/self.max_steps])
