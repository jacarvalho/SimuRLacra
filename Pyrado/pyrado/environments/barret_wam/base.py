import numpy as np
import robcom_python as robcom

import pyrado
from pyrado.environments.base import Env
from pyrado.spaces import BoxSpace
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


class WAMBallInCupReal(Env):
    """ Class for the real Barret WAM """

    name: str = 'wam-real'

    def __init__(self,
                 dt: float = 1 / 500.,
                 max_steps: int = pyrado.inf,
                 ip: str = '127.0.0.1',
                 poses_des: [np.ndarray, None] = None):
        """
        Constructor

        :param dt: sampling time interval
        :param max_steps: maximum number of time steps
        :param ip: IP address of the Qube platform
        :param poses_des: desired joint poses as ndarray of shape (max_steps, 3)
        """
        # Call the base class constructor to initialize fundamental members
        super().__init__(dt, max_steps)

        # Desired joint position for the initial state
        self.init_pose_des = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])

        # Desired trajectory (has to be shorter than or equal to max_steps)
        self.poses_des = poses_des

        # Connect to client
        self._client = robcom.Client()
        self._client.start(ip, 2013)  # ip adress and port
        print_cbt('Connected to the Barret WAM client.', 'c', bright=True)
        self._gt = None  # Goto command

        # Initialize spaces
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._create_spaces()

        # Initialize task
        self._task = self._create_task(dict())

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
        # No task used for the moment. TODO: Formulate proper task
        return None

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

        # Move to initial state in 5 seconds
        gt.add_step(5., self.init_pose_des)
        print_cbt('Moving the Barret WAM to the initial position.', 'c')

        # Start process and wait for completion
        gt.start()
        gt.wait_for_completion()
        print_cbt('Reached the initial position.', 'c')

        # Reset time steps
        self._curr_step = 0

        return self.observe(self.state)

    def step(self, act: np.ndarray) -> tuple:
        # zero step reward
        self._curr_rew = 0.
        done = False
        info = dict()

        act = self._limit_act(act)

        # Only use `act` if no desired trajectory is given
        if self.poses_des is None:
            des_qpos = self.init_pose_des.copy()
            np.add.at(des_qpos, [1, 3, 5], act[:3])
        else:
            des_qpos = self.poses_des[self.curr_step]

        # Create robcom GoTo process at the first time step
        if self._curr_step == 0:
            self._gt = self._client.create(robcom.Goto, 'RIGHT_ARM', '')

        # Add desired joint position as step to the process
        self._gt.add_step(self.dt, des_qpos)
        self._curr_step += 1

        # Only start execution of process when all desired poses have been added to process i.e. `max_steps` has been reached.
        if self._curr_step >= self._max_steps:
            done = True

        if done:
            print_cbt('Executing trajectory on Barret WAM.', 'c')
            self._gt.start()
            self._gt.wait_for_completion()
            print_cbt('Finished execution.', 'c')
            # Get episode reward as user input (commented out for the moment)
            # self._curr_rew = float(input('Enter episode reward: '))

        return self.observe(self.state), self._curr_rew, done, info

    def render(self, mode: RenderMode, render_step: int = 1):
        pass

    def close(self):
        self._client.close()
        print_cbt('Connection to WAM client closed.', 'c')

    def observe(self, state: np.ndarray) -> np.ndarray:
        # Only observe the normalized time
        return np.array([self._curr_step/self.max_steps])