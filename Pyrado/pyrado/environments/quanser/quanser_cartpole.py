import numpy as np
import time
from abc import abstractmethod
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.quanser.base import RealEnv
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import UnderActuatedSwingUpRewFcn, ExpQuadrErrRewFcn
from pyrado.utils.input_output import print_cbt


class QCartPoleReal(RealEnv, Serializable):
    """ Base class for the real Quanser Cart-Pole """

    def __init__(self,
                 dt: float = 1/500.,
                 max_steps: int = pyrado.inf,
                 ip: str = '10.0.0.3',
                 state_des: np.ndarray = None):
        """
        Constructor

        :param dt: time step size on the Quanser device [s]
        :param max_steps: maximum number of steps executed on the device
        :param ip: IP address of the Cart-Pole platform
        :param state_des: desired state for the task
        """
        Serializable._init(self, locals())

        # Initialize spaces, dt, max_step, and communication
        super().__init__(ip, rcv_dim=4, snd_dim=2, dt=dt, max_steps=max_steps, state_des=state_des)
        self._curr_act = np.zeros(self.act_space.shape)  # just for usage in render function

        # Calibration and limits
        self._l_rail = 0.814  # [m]
        self._x_buffer = 0.05  # [m]
        self._calibrated = False
        self._c_lim = 0.075
        self._norm_x_lim = np.zeros(2)

    def _create_spaces(self):
        # Define the spaces
        self._state_space = None  # needs to be set in subclasses
        max_obs = np.array([0.814/2., 1., 1., pyrado.inf, pyrado.inf])
        max_act = np.array([8.])  # [V], original: 24
        self._obs_space = BoxSpace(max_obs, max_obs,
                                   labels=['$x$', r'$\sin\theta$', r'$\cos\theta$', r'$\dot{x}$', r'$\dot{\theta}$'])
        self._act_space = BoxSpace(-max_act, max_act, labels=['$V$'])

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        # Needs to implemented by subclasses
        return NotImplementedError

    @property
    def task(self):
        return self._task

    def observe(self, state):
        return np.array([state[0], np.sin(state[1]), np.cos(state[1]), state[2], state[3]])

    def reset(self, *args, **kwargs):
        # Reset socket, task, and calibrate
        super().reset(args, kwargs)

    def step(self, act):
        info = dict(t=self._curr_step*self._dt, act_raw=act)

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, self._curr_step, remaining_steps)

        # Apply actuator limits
        act_lim = self._limit_act(act)
        self._curr_act = act_lim

        # Send actions and receive sensor measurements
        meas = self._qsoc.snd_rcv(act_lim)

        # Transform the relative cart position to [-0.4, +0.4]
        if self._calibrated:
            meas[0] = (meas[0] - self._norm_x_lim[0]) - 0.5*(self._norm_x_lim[1] - self._norm_x_lim[0])
        # Normalize the angle from -pi to +pi:
        meas[1] = np.mod(meas[1] + np.pi, 2*np.pi) - np.pi

        # Construct the state from the measurements
        self.state = meas

        self._curr_step += 1

        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        if done:
            # Add final reward if done
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info

    def calibrate(self):
        if self._calibrated:
            return
        print_cbt('Calibrate the cartpole ...', 'c')

        # Go to the left
        print('\tGo to the Left:\t\t\t', end='')

        obs, _, _, _ = self.step(np.zeros(self.act_space.shape))
        ctrl = GoToLimCtrl(obs, positive=True)

        while not ctrl.done:
            act = ctrl(obs)
            obs, _, _, _ = self.step(act)

        if ctrl.success:
            self._norm_x_lim[1] = obs[0]
            print('\u2713')

        else:
            print('\u274C ')
            raise RuntimeError('Going to the left limit failed.')

        # Go to the right
        print('Go to the right ...\t\t', end='')

        obs, _, _, _ = self.step(np.zeros(self.act_space.shape))
        ctrl = GoToLimCtrl(obs, positive=False)

        while not ctrl.done:
            act = ctrl(obs)
            obs, _, _, _ = self.step(act)

        if ctrl.success:
            self._norm_x_lim[0] = obs[0]
            print('\u2713')
        else:
            print('\u274C')
            raise RuntimeError('Going to the right limit failed.')

        # Activate the absolute cart position:
        self._calibrated = True

    def _center_cart(self):
        """ Move the cart to the center (x = 0). """
        # Initialize
        t_max, t0 = 10.0, time.time()
        obs, _, _, _ = self.step(np.zeros(self.act_space.shape))

        print('Centering the cart ...\t\t', end='')
        while (time.time() - t0) < t_max:
            act = -np.sign(obs[0])*1.5*np.ones(self.act_space.shape)
            obs, _, _, _ = self.step(act)

            if np.abs(obs[0]) <= self._c_lim/10.:
                break

        # Stop the cart
        obs, _, _, _ = self.step(np.zeros(self.act_space.shape))
        time.sleep(0.5)

        if np.abs(obs[0]) > self._c_lim:
            print('\u274C')
            time.sleep(0.1)
            raise RuntimeError(
                f'Centering of the cart failed: |x| = {np.abs(obs[0]):.2f} > {self._c_lim:.2f}')

        print('\u2713')


class QCartPoleStabReal(QCartPoleReal):
    """ Stabilization task on the real Quanser Cart-Pole """

    name: str = 'qcp-st'

    def __init__(self,
                 dt: float = 1/500.,
                 max_steps: int = pyrado.inf,
                 ip: str = '10.0.0.3',
                 state_des: np.ndarray = None):
        """
        Constructor

        :param dt: time step size on the Quanser device [s]
        :param max_steps: maximum number of steps executed on the device
        :param ip: IP address of the Cartpole platform
        :param state_des: desired state for the task
        """
        super().__init__(dt, max_steps, ip, state_des)

        # Define the task-specific state space
        stab_thold = 15/180.*np.pi  # threshold angle for the stabilization task to be a failure [rad]
        min_state = np.array([-self._l_rail/2. + self._x_buffer, np.pi - stab_thold, np.inf, np.inf])
        max_state = np.array([+self._l_rail/2. - self._x_buffer, np.pi + stab_thold, np.inf, np.inf])
        self._state_space = BoxSpace(min_state, max_state, labels=['$x$', r'$\theta$', r'$\dot{x}$', r'$\dot{\theta}$'])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', None)
        if state_des is None:
            state_des = np.array([0., np.pi, 0., 0.])
        Q = np.diag([1e-0, 5e-0, 1e-2, 1e-2])
        R = np.diag([1e-2])
        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    def _wait_for_upright_pole(self, verbose=False):
        if verbose:
            print('\tCentering the Pole:\t\t', end='')

        # Initialize
        t_max, t0 = 15.0, time.time()
        upright = False

        pos_th = np.array([self._c_lim, 2.*np.pi/180.])
        vel_th = 0.1*np.ones(2)
        th = np.hstack((pos_th, vel_th))

        # Wait until the pole is upright
        while (time.time() - t0) <= t_max:
            obs, _, _, _ = self.step(np.zeros(self.act_space.shape))
            time.sleep(1/550.)

            transformed_state = np.array(obs, copy=True)
            transformed_state[1] -= np.sign(transformed_state[1])*np.pi
            if np.all(np.abs(transformed_state) <= th):
                upright = True
                break

        if not upright:
            if verbose:
                print('\u274C')
            time.sleep(0.1)
            state_str = np.array2string(np.abs(obs), suppress_small=True, precision=2,
                                        formatter={'float_kind': lambda x: '{0:+05.2f}'.format(x)})
            th_str = np.array2string(th, suppress_small=True, precision=2,
                                     formatter={'float_kind': lambda x: '{0:+05.2f}'.format(x)})
            raise TimeoutError('The pole is not upright: {0} > {1}'.format(state_str, th_str))

        elif verbose:
            print('\u2713')

        return

    def reset(self, *args, **kwargs):
        # Reset socket, task, and calibrate
        super().reset(args, kwargs)

        # Wait until the human reset the pole properly
        self._wait_for_upright_pole(verbose=True)

        # Start with a zero action and get the first sensor measurements
        meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

        # Reset time counter
        self._curr_step = 0

        return self.observe(meas)


class QCartPoleSwingUpReal(QCartPoleReal):
    """ Swing-up task on the real Quanser Cart-Pole """

    name: str = 'qcp-su'

    def __init__(self,
                 dt: float = 1/500.,
                 max_steps: int = pyrado.inf,
                 ip: str = '10.0.0.3',
                 state_des: np.ndarray = None):
        """
        Constructor

        :param dt: time step size on the Quanser device [s]
        :param max_steps: maximum number of steps executed on the device
        :param ip: IP address of the Cartpole platform
        :param state_des: desired state for the task
        """
        super().__init__(dt, max_steps, ip, state_des)

        # Define the task-specific state space
        max_state = np.array([self._l_rail/2. - self._x_buffer, +4*np.pi, np.inf, np.inf])  # [m, rad, m/s, rad/s]
        min_state = np.array([self._l_rail/2. + self._x_buffer, -4*np.pi, np.inf, np.inf])  # [m, rad, m/s, rad/s]
        self._state_space = BoxSpace(min_state, max_state, labels=['$x$', r'$\theta$', r'$\dot{x}$', r'$\dot{\theta}$'])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', None)
        if state_des is None:
            state_des = np.array([0., np.pi, 0., 0.])
        return FinalRewTask(
            RadiallySymmDesStateTask(self.spec, state_des, UnderActuatedSwingUpRewFcn(), idcs=[1]),
            mode=FinalRewMode(state_dependent=True, always_negative=True)
        )

    def reset(self, *args, **kwargs):
        # Reset socket and task
        super().reset()

        # The system only needs to be calibrated once, as this is a bit time consuming
        self.calibrate()

        # Center the cart in the middle
        self._center_cart()

        # Start with a zero action and get the first sensor measurements
        meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

        # Reset time counter
        self._curr_step = 0

        return self.observe(meas)


class GoToLimCtrl:
    """ Controller for going to one of the joint limits (part of the calibration routine) """

    def __init__(self, init_state: np.ndarray, positive: bool = True):
        """
        Constructor

        :param init_state: initial state of the system
        :param positive: direction switch
        """
        self.done = False
        self.success = False
        self.x_init = init_state[0]
        self.x_lim = 0.0
        self.xd_max = 1e-4
        self.delta_x_min = 0.1
        self.sign = 1 if positive else -1
        self.u_max = self.sign*np.array([1.5])
        self._t_init = False
        self._t0 = None
        self._t_max = 10.0
        self._t_min = 2.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Go to joint limits by applying u_max and save limit value in th_lim.

        :param obs: observation from the environment
        :return: action
        """
        x, _, _, xd, _ = obs

        # Initialize time
        if not self._t_init:
            self._t0 = time.time()
            self._t_init = True

        # Compute voltage
        if (time.time() - self._t0) < self._t_min:
            # Go full speed before t_min
            u = self.u_max
        elif (time.time() - self._t0) > self._t_max:
            # Do nothing if t_max is elapsed
            u = np.zeros(1)
            self.success = False
            self.done = True
        elif np.abs(xd) < self.xd_max:  # and np.abs(x - self.x_init) > self.delta_x_min:
            # Do nothing i
            u = np.zeros(1)
            self.success = True
            self.done = True
        else:
            u = self.u_max

        return u
