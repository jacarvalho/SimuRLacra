import numpy as np
from init_args_serializer import Serializable

from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.base import Env
from pyrado.utils.data_types import EnvSpec
from pyrado.spaces.box import BoxSpace


class StateAugmentationWrapper(EnvWrapper, Serializable):
    """
    StateAugmentationWrapper

    Augments the observation of the wrapped environment by its physics configuration
    """

    def __init__(self,
                 wrapped_env: Env,
                 params=None,
                 fixed=False):
        """
        Constructor

        :param wrapped_env: The environment to be wrapped
        :param params: The parameters to include in the observation
        :param fixed: Fix the parameters
        """
        Serializable._init(self, locals())

        EnvWrapper.__init__(self, wrapped_env)
        if params is not None:
            self._params = params
        else:
            self._params = list(inner_env(self.wrapped_env).domain_param.keys())
        self._nominal = inner_env(self.wrapped_env).get_nominal_domain_param()
        self._nominal = np.array([self._nominal[k] for k in self._params])
        self.fixed = fixed

    def _params_as_tensor(self):
        if self.fixed:
            return self._nominal
        else:
            return np.array([inner_env(self.wrapped_env).domain_param[k] for k in self._params])

    @property
    def obs_space(self):
        outer_space = self.wrapped_env.obs_space
        augmented_space = BoxSpace(0.5 * self._nominal, 1.5 * self._nominal, [self._nominal.shape[0]], self._params)
        return BoxSpace.cat((outer_space, augmented_space))

    def step(self, act: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(act)
        params = self._params_as_tensor()
        obs = np.concatenate((obs, params))
        return obs, reward, done, info

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None):
        obs = self.wrapped_env.reset(init_state, domain_param)
        params = self._params_as_tensor()
        obs = np.concatenate((obs, params))
        return obs

    @property
    def mask(self):
        return np.concatenate((np.zeros(self.wrapped_env.obs_space.flat_dim), np.ones(len(self._params))))

    @property
    def offset(self):
        return self.wrapped_env.obs_space.flat_dim

    def set_param(self, params):
        newp = dict()
        for key, value in zip(self._params, params):
            newp[key] = value.item()
        inner_env(self.wrapped_env).domain_param = newp

    def set_adv(self, params):
        for key, value in zip(self._params, params):
            inner_env(self.wrapped_env).domain_param[key] = self._nominal[key] + value

    @property
    def nominal(self):
        return self._nominal
