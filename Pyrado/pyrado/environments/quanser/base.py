from abc import ABC, abstractmethod
import numpy as np

import pyrado
from pyrado.environments.base import Env
from pyrado.environments.quanser.quanser_common import QSocket
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


class RealEnv(Env, ABC):
    """ Base class of all real-world environments in Pyrado """

    def __init__(self,
                 ip: str,
                 rcv_dim: int,
                 snd_dim: int,
                 dt: float = 1 / 500.,
                 max_steps: int = pyrado.inf,
                 state_des: np.ndarray = None):
        """
        Constructor

        :param ip: IP address of the platform
        :param rcv_dim: number of dimensions of the sensor i.e. measurement signal (received from Simulink server)
        :param snd_dim: number of dimensions of the action command (send to Simulink server)
        :param dt: sampling time interval
        :param max_steps: maximum number of time steps
        :param state_des: desired state for the task
        """
        # Call the base class constructor to initialize fundamental members
        super().__init__(dt, max_steps)

        # Initialize the state since it is needed for the first time the step fcn is called (in the reset fcn)
        self.state = np.zeros(rcv_dim)

        # Create a socket for communicating with the Quanser devices
        self._qsoc = QSocket(ip, rcv_dim, snd_dim)

        # Initialize spaces
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._create_spaces()

        # Initialize task
        self._state_des = state_des
        self._task = self._create_task(state_des)

    def __del__(self):
        """ Finalizer forwards to close function. """
        self.close()

    def close(self):
        """ Sends a zero-step and closes the communication. """
        if self._qsoc.is_open():
            self.step(np.zeros(self.act_space.shape))
            self._qsoc.close()
            print_cbt('Closed the connection to the Quanser device.', 'c', bright=True)

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

    @abstractmethod
    def _create_spaces(self):
        """
        Create spaces based on the domain parameters.
        Should set the attributes `_state_space`, `_obs_space`, and `_act_space`.

        .. note::
            This function is called from the constructor.
        """

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Reset the environment.
        The base version (re-)opens the socket and resets the task.

        :param args: just for compatibility with SimEnv. All args can be ignored.
        :param kwargs: just for compatibility with SimEnv. All kwargs can be ignored.
        """
        # Cancel and re-open the connection to the socket
        self._qsoc.close()
        self._qsoc.open()
        print_cbt('Opened the connection to the Quanser device.', 'c', bright=True)

        # Reset the task
        self._task.reset(env_spec=self.spec)

    def render(self, mode: RenderMode = RenderMode(text=True), render_step: int = 1):
        """
        Visualize one time step of the real-world device.
        The base version prints to console when the state exceeds its boundaries.

        :param mode: render mode: console, video, or both
        :param render_step: interval for rendering
        """
        if self._curr_step % render_step == 0:
            if mode.text:
                print(f'step: {self._curr_step:4d}  |  '
                      f'in bounds: {self._state_space.contains(self.state):1d}  |  '
                      f'rew: {self._curr_rew:1.3f}  |  '
                      f'act: {self._curr_act}  |  '
                      f'next state: {self.state}')
            if mode:
                # Print out of bounds to console if the mode is not empty
                self.state_space.contains(self.state, verbose=True)
