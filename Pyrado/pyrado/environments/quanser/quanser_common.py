import numpy as np
import socket
import struct
from scipy import signal


class QSocket:
    """ Handles the communication with Quarc (TCP/IP connection) """

    def __init__(self, ip: str, x_len: int, u_len: int):
        """
        Prepare socket for communication.

        :param ip: IP address of the Windows PC
        :param x_len: number of measured state variables to receive
        :param u_len: number of control variables to send
        """
        self._x_fmt = '>' + x_len*'d'
        self._u_fmt = '>' + u_len*'d'
        self._buf_size = x_len*8  # 8 bytes for each double
        self._port = 9095  # see the server block in the Simulink models
        self._ip = ip
        self._soc = None

    def snd_rcv(self, u) -> np.ndarray:
        """
        Send u and receive x.

        :param u: control vector
        :return: x: vector of measured states
        """
        self._soc.send(struct.pack(self._u_fmt, *u))
        data = self._soc.recv(self._buf_size)
        return np.array(struct.unpack(self._x_fmt, data), dtype=np.float32)

    def open(self):
        if self._soc is None:
            self._soc = socket.socket()
            self._soc.connect((self._ip, self._port))

    def close(self):
        if self._soc is not None:
            self._soc.close()
            self._soc = None

    def is_open(self) -> bool:
        """ Return True is the socket connection os open and False if not. """
        return False if self._soc is None else True


class VelocityFilter:
    """
    Discrete velocity filter derived from a continuous one

    .. note::
        This velocity filter class is currently not used since we now get the velocities from the Simulink model.
    """

    def __init__(self, x_len: int, num: tuple = (50, 0), den: tuple = (1, 50), dt: float = 0.002,
                 x_init: np.ndarray = None):
        """
        Initialize discrete filter coefficients.

        :param x_len: number of measured state variables to receive
        :param num: continuous-time filter numerator (continuous time)
        :param den: continuous-time filter denominator (continuous time)
        :param dt: sampling time interval
        :param x_init: initial observation of the signal to filter
        """
        derivative_filter = signal.cont2discrete((num, den), dt)
        self.b = derivative_filter[0].ravel().astype(np.float32)
        self.a = derivative_filter[1].astype(np.float32)
        if x_init is None:
            self.z = np.zeros((max(len(self.a), len(self.b)) - 1, x_len), dtype=np.float32)
        else:
            self.set_initial_state(x_init)

    def set_initial_state(self, x_init: np.ndarray):
        """
        This method can be used to set the initial state of the velocity filter. This is useful when the initial
        (position) observation has been retrieved and it is non-zero.
        Otherwise the filter would assume a very high velocity.
        :param x_init: initial observation
        """
        assert isinstance(x_init, np.ndarray)

        # Get the initial condition of the filter
        zi = signal.lfilter_zi(self.b, self.a)  # dim = order of the filter = 1

        # Set the filter state
        self.z = np.outer(zi, x_init)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the filter.

        :param x: observed position signal
        :return x_dot: filtered velocity signal
        """
        x_dot, self.z = signal.lfilter(self.b, self.a, x[None, :], 0, self.z)
        return x_dot.ravel()
