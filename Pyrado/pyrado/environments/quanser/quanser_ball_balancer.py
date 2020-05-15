import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.quanser.base import RealEnv
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ScaledExpQuadrErrRewFcn


class QBallBalancerReal(RealEnv, Serializable):
    """ Class for the real Quanser Ball-Balancer """

    name: str = 'qbb'

    def __init__(self,
                 dt: float = 1/500.,
                 max_steps: int = pyrado.inf,
                 ip: str = '192.168.2.5',
                 state_des: np.ndarray = None):
        """
        Constructor

        :param dt: sampling frequency on the [Hz]
        :param max_steps: maximum number of steps executed on the device [-]
        :param ip: IP address of the 2 DOF Ball-Balancer platform
        :param state_des: desired state for the task
        """
        Serializable._init(self, locals())

        # Initialize spaces, dt, max_step, and communication
        super().__init__(ip, rcv_dim=8, snd_dim=2, dt=dt, max_steps=max_steps, state_des=state_des)
        self._curr_act = np.zeros(self.act_space.shape)  # just for usage in render function

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array([np.pi/4., np.pi/4., 0.275/2., 0.275/2.,  # [rad, rad, m, m, rad/s, ...
                              5*np.pi, 5*np.pi, 0.5, 0.5])  # ... rad/s, rad/s, m/s, m/s]
        max_act = np.array([3., 3.])  # [V]
        self._state_space = BoxSpace(-max_state, max_state,
                                     labels=[r'$\theta_{x}$', r'$\theta_{y}$', '$x$', '$y$',
                                             r'$\dot{\theta}_{x}$', r'$\dot{\theta}_{y}$', r'$\dot{x}$', r'$\dot{y}$'])
        self._obs_space = self._state_space
        self._act_space = BoxSpace(-max_act, max_act, labels=['V_{x}', 'V_{y}'])

    def _create_task(self, state_des: [np.ndarray, None]) -> Task:
        # Define the task including the reward function
        if state_des is None:
            state_des = np.zeros(self.state_space.shape)
        Q = np.diag([1e0, 1e0, 1e3, 1e3, 1e-2, 1e-2, 5e-1, 5e-1])
        R = np.diag([1e-2, 1e-2])
        return DesStateTask(
            self.spec, state_des, ScaledExpQuadrErrRewFcn(Q, R, self._state_space, self.act_space, min_rew=1e-4)
        )

    def reset(self, *args, **kwargs) -> np.ndarray:
        # Reset socket and task
        super().reset()

        # Start with a zero action and get the first sensor measurements
        meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

        # Reset time counter
        self._curr_step = 0

        return self.observe(meas)

    def step(self, act: np.ndarray) -> tuple:
        info = dict(t=self._curr_step*self._dt, act_raw=act)

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)

        # Apply actuator limits
        act_lim = self._limit_act(act)
        self._curr_act = act_lim

        # Send actions and receive sensor measurements
        meas = self._qsoc.snd_rcv(act_lim)

        # Construct the state from the measurements
        self.state = meas

        self._curr_step += 1

        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        if done:
            # Add final reward if done
            remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info
