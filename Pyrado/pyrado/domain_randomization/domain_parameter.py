import torch as to
from abc import ABC, abstractmethod
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from typing import Sequence

import pyrado


class DomainParam(ABC):
    """ Class to store and manage a (single) domain parameter a.k.a. physics parameter a.k.a. simulator parameter """

    def __init__(self,
                 name: str,
                 mean: [int, float, to.Tensor],
                 clip_lo: [int, float] = -pyrado.inf,
                 clip_up: [int, float] = pyrado.inf,
                 roundint=False):
        """
        Constructor, also see the constructor of DomainRandomizer.

        :param name: name of the parameter
        :param mean: nominal parameter value
        :param clip_lo: lower value for clipping
        :param clip_up: upper value for clipping
        :param roundint: flags if the parameters should be rounded and converted to an integer
        """
        self.name = name
        self.mean = mean
        self.clip_lo = clip_lo
        self.clip_up = clip_up
        self.roundint = roundint
        self.distr = None  # no randomization by default

    def __eq__(self, other):
        """ Check if two `DomainParam` are equal by comparing all attributes defined in `get_field_names()`. """
        if not isinstance(other, DomainParam):
            raise pyrado.TypeErr(given=other, expected_type=DomainParam)

        for fn in self.get_field_names():
            if getattr(self, fn) != getattr(other, fn):
                return False
        return True

    @staticmethod
    def get_field_names() -> Sequence[str]:
        """ Get union of all hyper-parameters of all domain parameter distributions. """
        raise NotImplementedError

    def adapt(self, domain_distr_param: str, domain_distr_param_value: [float, int]):
        """
        Update this domain parameter.

        .. note::
            This function should by called by the subclasses' `adapt()` function.

        :param domain_distr_param: distribution parameter to update, e.g. mean or std
        :param domain_distr_param_value: new value of the distribution parameter
        """
        if domain_distr_param not in self.get_field_names():
            raise KeyError(f'The domain parameter {self.name} does not have a domain distribution parameter'
                           f'called {domain_distr_param}!')
        setattr(self, domain_distr_param, domain_distr_param_value)

    def sample(self, num_samples: int = 1) -> list:
        """
        Generate new domain parameter values.

        :param num_samples: number of samples (sets of new parameter values)
        :return: list of Tensors containing the new parameter values
        """
        assert isinstance(num_samples, int) and num_samples > 0

        if self.distr is None:
            # Return nominal values multiple times
            return list(to.ones(num_samples) * self.mean)
        else:
            # Draw num_samples samples (rsample is not implemented for Bernoulli)
            sample_tensor = self.distr.sample(sample_shape=to.Size([num_samples]))

            # Clip the values
            sample_tensor = to.clamp(sample_tensor, self.clip_lo, self.clip_up)

            # Round values to integers if desired
            if self.roundint:
                sample_tensor = to.round(sample_tensor).type(to.int)

            # Convert the large tensor into a list of small tensors
            return list(sample_tensor)


class UniformDomainParam(DomainParam):
    """ Domain parameter sampled from a normal distribution """

    def __init__(self, halfspan: float, **kwargs):
        """
        Constructor

        :param halfspan: half interval (mean is already mandatory for super-class `DomainParam`)
        :param kwargs: forwarded to `DomainParam` constructor
        """
        super().__init__(**kwargs)

        self.halfspan = halfspan
        self.distr = Uniform(self.mean - self.halfspan, self.mean + self.halfspan)

    @staticmethod
    def get_field_names() -> Sequence[str]:
        return ['name', 'mean', 'halfspan', 'clip_lo', 'clip_up', 'roundint']

    def adapt(self, domain_distr_param: str, domain_distr_param_value: [float, int]):
        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distribution, otherwise the changes will have no effect
        self.distr = Uniform(self.mean - self.halfspan, self.mean + self.halfspan)


class NormalDomainParam(DomainParam):
    """ Domain parameter sampled from a normal distribution """

    def __init__(self, std: [float, to.Tensor], **kwargs):
        """
        Constructor

        :param std: standard deviation (mean is already mandatory for super-class `DomainParam`)
        :param kwargs: forwarded to `DomainParam` constructor
        """
        super().__init__(**kwargs)

        self.std = std
        self.distr = Normal(self.mean, self.std)

    @staticmethod
    def get_field_names() -> Sequence[str]:
        return ['name', 'mean', 'std', 'clip_lo', 'clip_up', 'roundint']

    def adapt(self, domain_distr_param: str, domain_distr_param_value: [float, int]):
        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distribution, otherwise the changes will have no effect
        self.distr = Normal(self.mean, self.std)


class MultivariateNormalDomainParam(DomainParam):
    """ Domain parameter sampled from a normal distribution """

    def __init__(self, cov: [to.Tensor], **kwargs):
        """
        Constructor

        :param cov: covariance (mean is already mandatory for super-class `DomainParam`)
        :param kwargs: forwarded to `DomainParam` constructor
        """
        assert len(cov.shape) == 2, 'Covariance needs to be given as a matrix'
        super().__init__(**kwargs)

        self.mean = self.mean.view(-1, )
        self.cov = cov
        self.distr = MultivariateNormal(self.mean, self.cov)

    @staticmethod
    def get_field_names() -> Sequence[str]:
        return ['name', 'mean', 'cov', 'clip_lo', 'clip_up', 'roundint']

    def adapt(self, domain_distr_param: str, domain_distr_param_value: [float, int]):
        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distribution, otherwise the changes will have no effect
        self.distr = MultivariateNormal(self.mean, self.cov)


class BernoulliDomainParam(DomainParam):
    """ Domain parameter sampled from a Bernoulli distribution """

    def __init__(self, val_0: [int, float], val_1: [int, float], prob_1: float, **kwargs):
        """
        Constructor

        :param val_0: value of event 0
        :param val_1: value of event 1
        :param prob_1: probability of event 1, equals 1 - probability of event 0
        :param kwargs: forwarded to `DomainParam` constructor
        """
        if 'mean' not in kwargs:
            kwargs['mean'] = None
        super().__init__(**kwargs)

        self.val_0 = val_0
        self.val_1 = val_1
        self.prob_1 = prob_1
        self.distr = Bernoulli(self.prob_1)

    @staticmethod
    def get_field_names() -> Sequence[str]:
        return ['name', 'mean', 'val_0', 'val_1', 'prob_1', 'clip_lo', 'clip_up', 'roundint']

    def adapt(self, domain_distr_param: str, domain_distr_param_value: [float, int]):
        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distribution, otherwise the changes will have no effect
        self.distr = Bernoulli(self.prob_1)

    def sample(self, num_samples: int = 1) -> list:
        """
        Generate new domain parameter values.

        :param num_samples: number of samples (sets of new parameter values)
        :return: list of Tensors containing the new parameter values
        """
        assert isinstance(num_samples, int) and num_samples > 0

        if self.distr is None:
            # Return nominal values multiple times
            return list(to.ones(num_samples) * self.mean)
        else:
            # Draw num_samples samples (rsample is not implemented for Bernoulli)
            sample_tensor = self.distr.sample(sample_shape=to.Size([num_samples]))

            # Sample_tensor contains either 0 or 1
            sample_tensor = (to.ones_like(sample_tensor) - sample_tensor) * self.val_0 + sample_tensor * self.val_1

            # Clip the values
            sample_tensor = to.clamp(sample_tensor, self.clip_lo, self.clip_up)

            # Round values to integers if desired
            if self.roundint:
                sample_tensor = to.round(sample_tensor).type(to.int)

            # Convert the large tensor into a list of small tensors
            return list(sample_tensor)
