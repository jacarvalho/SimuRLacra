import os.path as osp
import numpy as np
import robcom_python as robcom
from init_args_serializer import Serializable

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


class WAMBallInCupReal(Env, Serializable):
    """
    Class for the real Barrett WAM

    Uses robcom 2.0 and specifically robcom's GoTo process to execute a trajectory given by desired joint positions.
    The process is only executed on the real system after `max_steps` has been reached to avoid possible latency,
    but at the same time mimic the usual step-based environment behavior.
    """

    name: str = 'wam-bic'

    def __init__(self,
                 dt: float = 1/500.,
                 max_steps: int = pyrado.inf,
                 ip: [str, None] = '192.168.2.2'):
        """
        Constructor

        :param dt: sampling time interval
        :param max_steps: maximum number of time steps
        :param ip: IP address of the PC controlling the Barrett WAM, pass `None` to skip connecting
        :param save_trajectory: If the trajectory recorded from Barrett WAM should be saved
        """
        Serializable._init(self, locals())

        # Make sure max_steps is reachable
        if not max_steps < pyrado.inf:
            raise pyrado.ValueErr(given=max_steps, given_name='max_steps', l_constraint=pyrado.inf)

        # Call the base class constructor to initialize fundamental members
        super().__init__(dt, max_steps)

        # Create the robcom client and connect to it
        self._client = robcom.Client()
        if ip is not None:
            self._client.start(ip, 2013)  # IP address and port
            print_cbt('Connected to the Barret WAM client.', 'c', bright=True)
        self._dc = None  # Goto command

        # Desired joint position for the initial state
        self.init_pose_des = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])

        # Initialize params
        self._curr_step_rr = 0
        self.qpos_des = None
        self.qvel_des = None
        self.qpos = None
        self.qvel = None

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
        # The wrapped task acts as a dummy and carries the FinalRewTask
        return FinalRewTask(GoallessTask(self.spec, ZeroPerStepRewFcn()), mode=FinalRewMode(user_input=True))

    def _create_spaces(self):
        # State space (normalized time, since we do not have a simulation)
        self._state_space = BoxSpace(np.array([0.]), np.array([1.]))

        # Action space (PD controller on 3 joint positions and velocities)
        act_up = np.array([1.985, np.pi, np.pi/2, 10*np.pi, 10*np.pi, 10*np.pi])
        act_lo = np.array([-1.985, -0.9, -np.pi/2, -10*np.pi, -10*np.pi, -10*np.pi])
        self._act_space = BoxSpace(act_lo, act_up,  # [rad, rad, rad, rad/s, rad/s, rad/s]
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

        # Reset the task which also resets the reward function if necessary
        self._task.reset(env_spec=self.spec)

        # Reset time steps
        self._curr_step = 0
        self._curr_step_rr = 0
        self.state = np.array([self._curr_step/self.max_steps])

        # Reset trajectory params
        self.qpos_des = np.tile(self.init_pose_des, (self.max_steps, 1))
        self.qvel_des = np.zeros_like(self.qpos_des)
        self.qpos = np.zeros_like(self.qpos_des)
        self.qvel = np.zeros_like(self.qpos_des)

        input('Hit enter to continue.')

        # Create robcom direct-control process
        self._dc = self._client.create(robcom.ClosedLoopDirectControl, 'RIGHT_ARM', '')

        return self.observe(self.state)

    def step(self, act: np.ndarray) -> tuple:
        info = dict(t=self._curr_step*self._dt, act_raw=act)

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)  # always 0 for wam-bic-real

        # Limit the action
        act = self.limit_act(act)

        # the policy operates on joint 1, 3 and 5
        np.add.at(self.qpos_des[self._curr_step], [1, 3, 5], act[:3])
        np.add.at(self.qvel_des[self._curr_step], [1, 3, 5], act[3:])

        # Update current step and state
        self._curr_step += 1
        self.state = np.array([self._curr_step/self.max_steps])

        # A GoallessTask only signals done when has_failed() is true, i.e. the the state is out of bounds
        done = self._task.is_done(self.state)  # always false for wam-bic-real

        # Only start execution of process when all desired poses have been sampled from the policy
        # i.e. `max_steps` has been reached
        if self._curr_step >= self._max_steps:
            done = True
            print_cbt('Executing trajectory on Barret WAM.', 'c', bright=True)
            self._dc.start(False, 1, self._callback, ['POS', 'VEL'], [], [])
            self._dc.wait_for_completion()
            print_cbt('Finished execution.', 'c')

        # Add final reward if done
        if done:
            # Ask the user to enter the final reward
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info

    def _callback(self, jg, eg, data_provider):
        """
        This function is called from robcom as callback and should never be called manually

        :param jg: joint group
        :param eg: endeffector group
        :param data_provider: additional datastream
        """

        # Check if max_steps is reached
        if self._curr_step_rr >= self.max_steps:
            return True

        # Get current joint position and velocity
        self.qpos[self._curr_step_rr] = np.array(jg.get(robcom.JointState.POS))
        self.qvel[self._curr_step_rr] = np.array(jg.get(robcom.JointState.VEL))

        # Set desired joint position and velocity
        dpos = self.qpos_des[self._curr_step_rr].tolist()
        dvel = self.qvel_des[self._curr_step_rr].tolist()
        jg.set(robcom.JointDesState.POS, dpos)
        jg.set(robcom.JointDesState.VEL, dvel)

        # Update current step at real robot
        self._curr_step_rr += 1

        return False

    def render(self, mode: RenderMode, render_step: int = 1):
        # Skip all rendering
        pass

    def close(self):
        # Don't close the connection to robcom manually, since this might cause SL to crash.
        # Closing the connection is finally handled by robcom
        pass
