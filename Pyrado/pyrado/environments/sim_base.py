from abc import ABC, abstractmethod
from collections import Iterable
from init_args_serializer import Serializable

from pyrado.environments.base import Env
from pyrado.utils.data_types import RenderMode
from pyrado.spaces.base import Space


class SimEnv(Env, ABC, Serializable):
    """
    Base class of all simulated environments in Pyrado.
    Uses Serializable to facilitate proper serialization.
    The domain parameters are automatically part of the serialized state.
    """

    @property
    @abstractmethod
    def init_space(self) -> Space:
        """ Get the initial state space. """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_nominal_domain_param(cls) -> dict:
        """
        Get the nominal a.k.a. default domain parameters.

        .. note::
            This function is used to check which domain parameters exist.
        """
        raise NotImplementedError

    @property
    def supported_domain_param(self) -> Iterable:
        """
        Get an iterable of all supported domain parameters.
        The default implementation takes the keys of `get_nominal_domain_param()`.
        The domain parameters are automatically stored in attributes prefixed with '_'.
        """
        return self.get_nominal_domain_param().keys()

    @property
    @abstractmethod
    def domain_param(self) -> dict:
        """
        Get the environment's domain parameters.
        The domain parameters are synonymous to the parameters used by the simulator to run the physics simulation
        (e.g., masses, extents, or friction coefficients). This must include all parameters that can be randomized,
        but there might also be additional parameters that depend on the domain parameters.
        """
        raise NotImplementedError

    @domain_param.setter
    @abstractmethod
    def domain_param(self, param: dict):
        """
        Set the environment's domain parameters.
        The changes are only applied at the next call of the reset function.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, mode: RenderMode, render_step: int = 1):
        """
        Visualize one time step of the simulation.
        The base version prints to console when the state exceeds its boundaries.

        :param mode: render mode: console, video, or both
        :param render_step: interval for rendering
        """
        # Print to console if the mode is not empty
        if mode.text or mode.video:
            self.state_space.contains(self.state, verbose=True)

    def close(self):
        """ For compatibility to `RealEnv` with the wrappers which are subclasses of `Env`. """
        pass

    def _get_state(self, state_dict: dict):
        super(SimEnv, self)._get_state(state_dict)
        # Add
        state_dict['domain_param'] = self.domain_param

    def _set_state(self, state_dict: dict, copying: bool = False):
        super(SimEnv, self)._set_state(state_dict, copying=copying)
        # Retrieve
        self.domain_param = state_dict['domain_param']
