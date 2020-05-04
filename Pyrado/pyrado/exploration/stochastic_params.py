import torch as to
from abc import ABC, abstractmethod
from typing import Sequence
from warnings import warn

import pyrado
from pyrado.exploration.normal_noise import DiagNormalNoise, FullNormalNoise
from pyrado.utils.properties import Delegate
from pyrado.sampling.hyper_sphere import sample_from_hyper_sphere_surface


class StochasticParamExplStrat(ABC):
    """ Exploration strategy which samples policy parameters from a distribution. """

    def __init__(self, param_dim: int):
        """
        Constructor

        :param param_dim: number of policy parameters
        """
        self.param_dim = param_dim

    @abstractmethod
    def sample_param_set(self, nominal_params: to.Tensor) -> to.Tensor:
        """
        Sample one set of policy parameters from the current distribution.

        :param nominal_params: parameter set (1-dim tensor) to sample around
        :return: sampled parmeter set (1-dim tensor)
        """
        raise NotImplementedError

    def sample_param_sets(self,
                          nominal_params: to.Tensor,
                          num_samples: int,
                          include_nominal_params: bool = False) -> to.Tensor:
        """ 
        Sample multiple sets of policy parameters from the current distribution.

        :param nominal_params: parameter set (1-dim tensor) to sample around
        :param num_samples: number of parameter sets
        :param include_nominal_params: `True` to include the nominal parameter values as first parameter set
        :return: policy parameter sets as NxP or (N+1)xP tensor where N is the number samples and P is the number of
                 policy parameters
        """
        ps = [self.sample_param_set(nominal_params) for _ in range(num_samples)]
        if include_nominal_params:
            ps.insert(0, nominal_params)
        return to.stack(ps)


class SymmParamExplStrat(StochasticParamExplStrat):
    """
    Wrap a parameter exploration strategy to enforce symmetric sampling.
    The function `sample_param_sets` will always return an even number of parameters, and it's guaranteed that
    ps[:len(ps)//2] == -ps[len(ps)//2:]
    """

    def __init__(self, wrapped: StochasticParamExplStrat):
        """
        Constructor

        :param wrapped: exploration strategy to wrap around
        """
        super().__init__(wrapped.param_dim)
        self.wrapped = wrapped

    def sample_param_set(self, nominal_params: to.Tensor) -> to.Tensor:
        # Should not be done, but fail gracefully
        warn('Called sample_param_set on SymmParamExplStrat, which will still return only one param set', stacklevel=2)
        return self.wrapped.sample_param_set(nominal_params)

    def sample_param_sets(self, nominal_params: to.Tensor,
                          num_samples: int,
                          include_nominal_params: bool = False) -> to.Tensor:
        # Adjust sample size to be even
        if num_samples % 2 != 0:
            num_samples += 1

        # Sample one half
        pos_half = self.wrapped.sample_param_sets(nominal_params, num_samples // 2)

        # Mirror around nominal params for the other half
        # hp = nom + eps => eps = hp - nom
        # hn = nom - eps = nom - (hp - nom) = 2*nom - hp
        neg_half = 2. * nominal_params - pos_half
        parts = [pos_half, neg_half]

        # Add nominal params if requested
        if include_nominal_params:
            parts.insert(0, nominal_params.view(1, -1))
        return to.cat(parts)

    def __getattr__(self, name: str):
        """
        Forward unknown attributes to wrapped strategy.

        :param name: name of the attribute
        """
        # getattr raises an AttributeError if not found, just as this method should.
        return getattr(self.wrapped, name)


class NormalParamNoise(StochasticParamExplStrat):
    """ Sample parameters from a normal distribution. """

    def __init__(self,
                 param_dim: int,
                 full_cov: bool = False,
                 std_init: float = 1.,
                 std_min: [float, Sequence[float]] = 0.01,
                 train_mean: bool = False):
        """
        Constructor

        :param param_dim: number of policy parameters
        :param full_cov: use a full covariance matrix or a diagonal covariance matrix (independent random variables)
        :param std_init: initial standard deviation for the noise distribution
        :param std_min: minimal standard deviation for the exploration noise
        :param train_mean: set `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        """
        # Call the StochasticParamExplStrat's constructor
        super().__init__(param_dim)

        if full_cov:
            self._noise = FullNormalNoise(noise_dim=param_dim, std_init=std_init, std_min=std_min,
                                          train_mean=train_mean)
        else:
            self._noise = DiagNormalNoise(noise_dim=param_dim, std_init=std_init, std_min=std_min,
                                          train_mean=train_mean)

    @property
    def noise(self) -> [FullNormalNoise, DiagNormalNoise]:
        """ Get the exploation noise. """
        return self._noise

    def sample_param_set(self, nominal_params: to.Tensor) -> to.Tensor:
        return self._noise(nominal_params).sample()

    def sample_param_sets(self, nominal_params: to.Tensor, num_samples: int, include_nominal_params: bool = False) -> to.Tensor:
        ps = self._noise(nominal_params).sample((num_samples,))
        if include_nominal_params:
            ps = to.cat([nominal_params.view(1, -1), ps])
        return ps

    # Make NormalParamNoise appear as if it would have the following functions / properties
    reset_expl_params = Delegate('_noise')
    adapt = Delegate('_noise')
    std = Delegate('_noise')
    cov = Delegate('_noise')
    get_entropy = Delegate('_noise')


class HyperSphereParamNoise(StochasticParamExplStrat):
    """ Sample parameters from a normal distribution. """

    def __init__(self,
                 param_dim: int,
                 expl_r_init: float = 1.0):
        """
        Constructor

        :param param_dim: number of policy parameters
        :param expl_r_init: initial radius of the hyper-sphere
        """
        # Call the base class constructor
        super().__init__(param_dim)

        self._r_init = expl_r_init
        self._r = expl_r_init

    @property
    def r(self) -> float:
        """ Get the radius of the hypersphere. """
        return self._r

    def reset_expl_params(self):
        """ Reset all parameters of the exploration strategy. """
        self._r = self._r_init

    def adapt(self, r: float):
        """ Set a new radius for the hyper sphere from which the policy parameters are sampled. """
        if not r > 0.0:
            pyrado.ValueErr(given=r, g_constraint='0')
        self._r = r

    def sample_param_set(self, nominal_params: to.Tensor) -> to.Tensor:
        return nominal_params + self._r * sample_from_hyper_sphere_surface(self.param_dim, 'normal')
