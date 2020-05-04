import numpy as np
from init_args_serializer import Serializable

from pyrado.environment_wrappers.base import EnvWrapperAct
from pyrado.environments.base import Env


class ActDelayWrapper(EnvWrapperAct, Serializable):
    """ Environment wrapper which delays actions by a fixed number of time steps. """

    def __init__(self, wrapped_env: Env, delay: int = 0):
        """
        Constructor

        :param wrapped_env: environment to wrap around (only makes sense from simulation environments)
        :param delay: integer action delay measured in number of time steps
        """
        Serializable._init(self, locals())

        # Invoke base constructor
        super().__init__(wrapped_env)

        # Store parameter and initialize slot for queue
        self._delay = delay
        self._act_queue = []

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay: int):
        # Validate and set
        assert isinstance(delay, int) and delay >= 0
        self._delay = delay

    def _save_domain_param(self, domain_param: dict):
        """
        Store the action delay in the domain parameter dict

        :param domain_param: domain parameter dict
        """
        # Cast to integer for consistency
        domain_param['act_delay'] = int(self._delay)

    def _load_domain_param(self, domain_param: dict):
        """
        Load the action delay from the domain parameter dict

        :param domain_param: domain parameter dict
        """
        # Cast the delay value to int, since randomizer yields ndarrays or Tensors
        self._delay = int(domain_param.get('act_delay', self._delay))

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None):
        # Adapt _delay to the new act_delay if provided
        if domain_param is not None:
            self._load_domain_param(domain_param)

        # Init action queue with the right amount of 0 actions
        self._act_queue = [np.zeros(self.act_space.shape)] * self._delay

        # Call the reset function of the super class and forwards the arguments
        return super().reset(init_state, domain_param)

    def _process_act(self, act: np.ndarray) -> np.ndarray:
        """
        Return the delayed action.

        :param act: commanded action which will be delayed by _delay time steps
        :return: next action that has been commanded _delay time steps before
        """
        if self._delay != 0:
            # Append current action to queue
            self._act_queue.append(act)

            # Retrieve and remove first element
            act = self._act_queue.pop(0)

        # Return modified action
        return act
