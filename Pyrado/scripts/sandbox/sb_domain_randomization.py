import torch as to

from pyrado.domain_randomization.domain_parameter import DomainParam, BernoulliDomainParam, UniformDomainParam,\
    NormalDomainParam, MultivariateNormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer


DomainParam(name='a', mean=1)

BernoulliDomainParam(name='b', val_0=2, val_1=5, prob_1=0.8)

DomainRandomizer(
    NormalDomainParam(name='mass', mean=1.2, std=0.1, clip_lo=10, clip_up=100)
)

DomainRandomizer(
    NormalDomainParam(name='mass', mean=1.2, std=0.1, clip_lo=10, clip_up=100),
    UniformDomainParam(name='special', mean=0, halfspan=42, clip_lo=-7.4, roundint=True),
    NormalDomainParam(name='length', mean=4, std=0.6, clip_up=50.1),
    UniformDomainParam(name='time_delay', mean=13, halfspan=6, clip_up=17, roundint=True),
    MultivariateNormalDomainParam(name='multidim', mean=10 * to.ones((2,)), cov=2*to.eye(2), clip_up=11)
)