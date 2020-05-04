import numpy as np
from init_args_serializer.serializable import Serializable

from pyrado.environment_wrappers.base import EnvWrapperObs
from pyrado.environments.sim_base import SimEnv


class GaussianObsNoiseWrapper(EnvWrapperObs, Serializable):
    """ Environment wrapper which adds normally distributed i.i.d. noise to all observations. """

    def __init__(self, wrapped_env: SimEnv, noise_mean: list = None, noise_std: list = None):
        """
        :param wrapped_env: environment to wrap

        :param noise_mean: list or ndarray for the mean of the noise (mostly all zeros)
        :param noise_std: list or ndarray for the standard deviation of the noise (no default value!)
        """
        Serializable._init(self, locals())

        super().__init__(wrapped_env)

        # Parse noise specification
        if noise_mean is not None:
            self._mean = np.array(noise_mean)
            assert self._mean.shape == self.obs_space.shape
        else:
            self._mean = np.zeros(self.obs_space.shape)
        if noise_std is not None:
            self._std = np.array(noise_std)
            assert self._std.shape == self.obs_space.shape
        else:
            self._std = np.zeros(self.obs_space.shape)

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        # Generate gaussian noise sample
        noise = np.random.randn(*self.obs_space.shape) * self._std + self._mean  # * to unsqueeze the tuple

        # Add it to the observation
        return obs + noise

    def _save_domain_param(self, domain_param: dict):
        """
        Store the observation noise parameters in the domain parameter dict

        :param domain_param: domain parameter dict
        """
        domain_param['obs_noise_mean'] = self._mean
        domain_param['obs_noise_std'] = self._std

    def _load_domain_param(self, domain_param: dict):
        """
        Load the observation noise parameters from the domain parameter dict

        :param domain_param: domain parameter dict
        """
        if 'obs_noise_mean' in domain_param:
            self._mean = np.array(domain_param['obs_noise_mean'])
            assert self._mean.shape == self.obs_space.shape
        if 'obs_noise_std' in domain_param:
            self._std = np.array(domain_param['obs_noise_std'])
            assert self._std.shape == self.obs_space.shape
