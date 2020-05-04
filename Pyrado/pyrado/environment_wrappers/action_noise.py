import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.base import Env
from pyrado.environment_wrappers.base import EnvWrapperAct


class GaussianActNoiseWrapper(EnvWrapperAct, Serializable):
    """
    Environment wrapper which adds normally distributed i.i.d. noise to all action.
    This noise is independent for the potentially applied action-based exploration strategy.
    """

    def __init__(self, wrapped_env: Env, noise_mean: np.ndarray = None, noise_std: np.ndarray = None):
        """
        Constructor

        :param wrapped_env: environment to wrap around (only makes sense from simulation environments)
        :param noise_mean: mean of the noise distribution
        :param noise_std: standard deviation of the noise distribution
        """
        Serializable._init(self, locals())

        # Invoke base constructor
        super().__init__(wrapped_env)

        # Parse noise specification
        if noise_mean is not None:
            self._mean = np.array(noise_mean)
            if not self._mean.shape == self.act_space.shape:
                raise pyrado.ShapeErr(given=self._mean, expected_match=self.act_space)
        else:
            self._mean = np.zeros(self.act_space.shape)
        if noise_std is not None:
            self._std = np.array(noise_std)
            if not self._std.shape == self.act_space.shape:
                raise pyrado.ShapeErr(given=self._noise_std, expected_match=self.act_space)
        else:
            self._std = np.zeros(self.act_space.shape)

    def _process_act(self, act: np.ndarray) -> np.ndarray:
        # Generate gaussian noise values
        noise = np.random.randn(*self.act_space.shape) * self._std + self._mean  # * to unsqueeze the tuple

        # Add it to the action
        return act + noise

    def _save_domain_param(self, domain_param: dict):
        """
        Store the action noise parameters in the domain parameter dict

        :param domain_param: domain parameter dict
        """
        domain_param['act_noise_mean'] = self._mean
        domain_param['act_noise_std'] = self._std

    def _load_domain_param(self, domain_param: dict):
        """
        Load the action noise parameters from the domain parameter dict

        :param domain_param: domain parameter dict
        """
        if 'act_noise_mean' in domain_param:
            self._noise_mean = np.array(domain_param['act_noise_mean'])
            if not self._noise_mean.shape == self.act_space.shape:
                raise pyrado.ShapeErr(given=self._noise_mean, expected_match=self.act_space)
        if 'act_noise_std' in domain_param:
            self._noise_std = np.array(domain_param['act_noise_std'])
            if not self._noise_std.shape == self.act_space.shape:
                raise pyrado.ShapeErr(given=self._noise_std, expected_match=self.act_space)
