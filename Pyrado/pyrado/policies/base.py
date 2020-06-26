from abc import ABC, abstractmethod
from typing import Callable

import torch as to
import torch.nn as nn
import torch.nn.utils.convert_parameters as cp
from torch.jit import ScriptModule, trace, script

import pyrado
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


def _get_or_create_grad(t):
    """
    Get the grad tensor for the given tensor, or create if missing.

    :param t: input tensor
    :return g: gradient attribute of input tensor, zeros if created
    """
    g = t.grad
    if g is None:
        g = to.zeros_like(t)
        t.grad = g
    return g


class Policy(nn.Module, ABC):
    """ Base class for all policies in Pyrado """

    def __init__(self, spec: EnvSpec, use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__()
        if not isinstance(spec, EnvSpec):
            raise pyrado.TypeErr(given=spec, expected_type=EnvSpec)

        self._env_spec = spec
        self._device = 'cuda' if use_cuda and to.cuda.is_available() else 'cpu'

    @property
    def device(self) -> str:
        """ Get the device (CPU or GPU) on which the policy is stored. """
        return self._device

    @property
    def env_spec(self) -> EnvSpec:
        """ Get the specification of environment the policy acts in. """
        return self._env_spec

    @property
    def param_values(self) -> to.Tensor:
        """
        Get the parameters of the policy as 1d array.
        The values are copied, modifying the return value does not propagate to the actual policy parameters.
        However, setting this variable will change the parameters.
        """
        return cp.parameters_to_vector(self.parameters())

    @param_values.setter
    def param_values(self, param: to.Tensor):
        """ Set the policy parameters from an 1d array. """
        if not self.param_values.shape == param.shape:
            raise pyrado.ShapeErr(given=param, expected_match=self.param_values)
        cp.vector_to_parameters(param, self.parameters())

    @property
    def param_grad(self) -> to.Tensor:
        """
        Get the gradient of the parameters as 1d array.
        The values are copied, modifying the return value does not propagate to the actual policy parameters.
        However, setting this variable will change the gradient.
        """
        return cp.parameters_to_vector(_get_or_create_grad(p) for p in self.parameters())

    @param_grad.setter
    def param_grad(self, param):
        """ Set the policy parameter gradient from an 1d array. """
        cp.vector_to_parameters(param, (_get_or_create_grad(p) for p in self.parameters()))

    @property
    def num_param(self) -> int:
        """ Get the number of policy parameters. """
        return sum(p.data.numel() for p in self.parameters())

    @property
    def is_recurrent(self) -> bool:
        """ Bool to signalise it the policy has a recurrent architecture. """
        return False

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        """
        Provide initial values for the hidden parameters. This should usually be a zero tensor.
        The default implementation will raise an error, to enforce override this function for recurrent policies.

        :param batch_size: number of states to track in parallel
        :return: Tensor of batch_size x hidden_size
        """
        raise AttributeError('Only recurrent policies should use the init_hidden() method.'
                             'Make sure to implementthis function for every recurrent policy type.')

    @abstractmethod
    def init_param(self, init_values: to.Tensor = None, **kwargs):
        """
        Initialize the policy's parameters. By default the parameters are initialized randomly.

        :param init_values: tensor of fixed initial policy parameter values
        :param kwargs: additional keyword arguments for the policy parameter initialization
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the policy to it's initial state.
        This should be called at the start of a rollout. Stateful policies should use it to reset the state variables.
        The default implementation does nothing.
        """
        pass  # this is used in rollout() even though PyCharm doesn't link it.

    @abstractmethod
    def forward(self, obs: to.Tensor) -> [to.Tensor, (to.Tensor, to.Tensor)]:
        """
        Get the action according to the policy and the observations (forward pass).

        :param obs: observation from the environment
        :return act: action to be taken
        """
        raise NotImplementedError

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = 'hidden_states') -> to.Tensor:
        """
        Re-evaluate the given rollout and return a derivable action tensor.
        The default implementation simply calls `forward()`.

        :param rollout: recorded, complete rollout
        :param hidden_states_name: name of hidden states rollout entry, used for recurrent networks.
                                   Defaults to 'hidden_states'. Change for value functions.
        :return: actions with gradient data
        """
        self.eval()
        return self(rollout.get_data_values('observations', truncate_last=True))  # all observations at once

    def trace(self) -> ScriptModule:
        """
        Create a ScriptModule from this policy.
        The returned module will always have the signature `action = tm(observation)`.
        For recurrent networks, it returns a stateful module that keeps the hidden states internally.
        Such modules have a reset() method to reset the hidden states.
        """
        # This does not work for recurrent policies, which is why they override this function.
        return script(TracedPolicyWrapper(self))


class TracedPolicyWrapper(nn.Module):
    """ Wrapper for a traced policy. Mainly used to add `input_size` and `output_size` attributes. """

    # Attributes
    input_size: int
    output_size: int

    def __init__(self, net: Policy):
        """
        Constructor

        :param net: non-recurrent network to wrap, which must not be a script module
        """
        super().__init__()

        # Setup attributes
        self.input_size = net.env_spec.obs_space.flat_dim
        self.output_size = net.env_spec.act_space.flat_dim

        self.net = trace(net, (to.from_numpy(net.env_spec.obs_space.sample_uniform()),))

    def forward(self, obs):
        return self.net(obs)


class ScaleLayer(nn.Module):
    """ Layer which scales the output of the input using a learnable scaling factor """

    def __init__(self, in_features: int, init_weight: float = 1.):
        """
        Constructor

        :param in_features: size of each input sample
        :param init_weight: initial scaling factor
        """
        super().__init__()
        self.weight = nn.Parameter(init_weight*to.ones(in_features, dtype=to.get_default_dtype()), requires_grad=True)

    def forward(self, inp: to.Tensor) -> to.Tensor:
        # Element-wise product
        return inp*self.weight


class PositiveScaleLayer(nn.Module):
    """ Layer which scales (strictly positive) the input using a learnable scaling factor """

    def __init__(self, in_features: int, init_weight: float = 1.):
        """
        Constructor

        :param in_features: size of each input sample
        :param init_weight: initial scaling factor
        """
        if not init_weight > 0:
            raise pyrado.ValueErr(given=init_weight, g_constraint='0')

        super().__init__()
        self.log_weight = nn.Parameter(to.log(init_weight*to.ones(in_features, dtype=to.get_default_dtype())),
                                       requires_grad=True)

    def forward(self, inp: to.Tensor) -> to.Tensor:
        # Element-wise product
        return inp*to.exp(self.log_weight)


class IndiNonlinLayer(nn.Module):
    """
    Layer subtracts a bias from the input, multiplies the result with a strictly positive sacaling factor, and then
    applies the provided nonlinearity. The scaling and the bias are learnable parameters.
    """

    def __init__(self,
                 in_features: int,
                 nonlin: Callable,
                 bias: bool,
                 weight: bool = True,
                 init_weight: float = 1.,
                 init_bias: float = 0.):
        """
        Constructor

        :param in_features: size of each input sample
        :param nonlin: nonlinearity
        :param bias: if `True`, a learnable bias is subtracted, else no bias is used
        :param weight: if `True` (default), the input is multiplied with a learnable scaling factor
        :param init_weight: initial scaling factor
        :param init_bias: initial bias
        """
        if not init_weight > 0:
            raise pyrado.ValueErr(given=init_weight, g_constraint='0')

        super().__init__()
        self._nonlin = nonlin
        if weight:
            self.log_weight = nn.Parameter(to.log(init_weight*to.ones(in_features, dtype=to.get_default_dtype())),
                                           requires_grad=True)
        else:
            self.log_weight = None
        if bias:
            self.bias = nn.Parameter(init_bias*to.ones(in_features, dtype=to.get_default_dtype()), requires_grad=True)
        else:
            self.bias = None

    def forward(self, inp: to.Tensor) -> to.Tensor:
        # Apply bias if desired
        tmp = inp - self.bias if self.bias is not None else inp
        # Apply weights if desired
        tmp = to.exp(self.log_weight)*tmp if self.log_weight is not None else tmp

        # y = f_nlin( w * (x-b) )
        return self._nonlin(tmp)
