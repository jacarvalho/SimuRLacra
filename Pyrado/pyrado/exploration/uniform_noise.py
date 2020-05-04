import torch as to
from torch.distributions import Uniform

import pyrado


class UniformNoise(to.nn.Module):
    """ Module for learnable additive uniform noise """

    def __init__(self,
                 noise_dim: [int, tuple],
                 halfspan_init: [float, to.Tensor],
                 halfspan_min: [float, to.Tensor] = 0.01,
                 train_mean: bool = False,
                 learnable: bool = True):
        """
        Constructor

        :param noise_dim: number of dimension
        :param halfspan_init: initial value of the half interval for the exploration noise
        :param halfspan_min: minimal value of the half interval for the exploration noise
        :param train_mean: `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param learnable: `True` if the parameters should be tuneable (default), `False` for shallow use (just sampling)
        """
        if not isinstance(halfspan_init, (float, to.Tensor)):
            raise pyrado.TypeErr(given=halfspan_init, expected_type=[float, to.Tensor])
        if not (isinstance(halfspan_init, float) and halfspan_init > 0 or
                isinstance(halfspan_init, to.Tensor) and all(halfspan_init > 0)):
            raise pyrado.ValueErr(given=halfspan_init, g_constraint='0')
        if not isinstance(halfspan_min, (float, to.Tensor)):
            raise pyrado.TypeErr(given=halfspan_min, expected_type=[float, to.Tensor])
        if not (isinstance(halfspan_min, float) and halfspan_min > 0 or
                isinstance(halfspan_min, to.Tensor) and all(halfspan_min > 0)):
            raise pyrado.ValueErr(given=halfspan_min, g_constraint='0')

        super().__init__()

        # Register parameters
        if learnable:
            self.log_halfspan = to.nn.Parameter(to.Tensor(noise_dim), requires_grad=True)
            self.mean = to.nn.Parameter(to.Tensor(noise_dim), requires_grad=True) if train_mean else None
        else:
            self.log_halfspan = to.empty(noise_dim)
            self.mean = None

        # Initialize parameters
        self.log_halfspan_init = to.log(to.tensor(halfspan_init)) if isinstance(halfspan_init, float) else to.log(halfspan_init)
        self.halfspan_min = to.tensor(halfspan_min) if isinstance(halfspan_min, float) else halfspan_min
        if not isinstance(self.log_halfspan_init, to.Tensor):
            raise pyrado.TypeErr(given=self.log_halfspan_init, expected_type=to.Tensor)
        if not isinstance(self.halfspan_min, to.Tensor):
            raise pyrado.TypeErr(given=self.halfspan_min, expected_type=to.Tensor)

        self.reset_expl_params()

    @property
    def halfspan(self) -> to.Tensor:
        """ Get the untransformed standard deviation vector given the log-transformed. """
        return to.max(to.exp(self.log_halfspan), self.halfspan_min)

    @halfspan.setter
    def halfspan(self, halfspan: to.Tensor):
        """
        Set the log-transformed half interval vector given the untransformed.
        This is useful if the `halfspan` should be a parameter for the optimizer, since the optimizer could set invalid
        values for the half interval span.
        """
        self.log_halfspan.data = to.log(to.max(halfspan, self.halfspan_min))

    def reset_expl_params(self):
        """ Reset all parameters of the exploration strategy. """
        if self.mean is not None:
            self.mean.data.zero_()
        self.log_halfspan.data.copy_(self.log_halfspan_init)

    def adapt(self, mean: to.Tensor = None, halfspan: [to.Tensor, float] = None):
        """
        Adapt the mean and the half interval span of the noise on the action or parameters.
        Use `None` to leave one of the parameters at their current value.

        :param mean: exploration strategy's new mean
        :param halfspan: exploration strategy's new half interval span
        """
        if not (isinstance(mean, to.Tensor) or mean is None):
            raise pyrado.TypeErr(given=mean, expected_type=to.Tensor)
        if not (isinstance(halfspan, to.Tensor) and (halfspan >= 0).all() or halfspan is None):

            raise pyrado.TypeErr(msg='The halfspan must be a Tensor with all elements > 0 or None!')
        if mean is not None:
            assert self.mean is not None, 'Can not change fixed zero mean!'
            if not mean.shape == self.mean.shape:
                raise pyrado.ShapeErr(given=mean, expected_match=self.mean)
            self.mean.data = mean
        if halfspan is not None:
            if not halfspan.shape == self.log_halfspan.shape:
                raise pyrado.ShapeErr(given=halfspan, expected_match=self.halfspan)
            self.halfspan = halfspan

    def forward(self, value: to.Tensor) -> Uniform:
        """
        Return the noise distribution for a specific noise-free value.

        :param value: value to evaluate the distribution around
        :return: noise distribution
        """
        mean = value if self.mean is None else value + self.mean
        return Uniform(low=mean-self.halfspan, high=mean+self.halfspan)

    def get_entropy(self) -> to.Tensor:
        """
        Get the exploration distribution's entropy.
        The entropy of a uniform distribution is independent of the mean.

        :return: entropy value
        """
        return to.log(2*self.halfspan)
