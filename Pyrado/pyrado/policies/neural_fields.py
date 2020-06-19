import torch as to
import torch.nn as nn
from typing import Callable

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy
from pyrado.policies.base_recurrent import RecurrentPolicy
from pyrado.policies.initialization import init_param
from pyrado.utils.input_output import print_cbt


class NFPolicy(RecurrentPolicy):
    """
    Neural Fields (NF)
    
    .. seealso::
        [1] S.-I. Amari "Dynamics of Pattern Formation in Lateral-Inhibition Type Neural Fields",
        Biological Cybernetics, 1977
    """

    def __init__(self,
                 spec: EnvSpec,
                 dt: float,
                 hidden_size: int,
                 conv_out_channels: int,
                 conv_kernel_size: int,
                 activation_nonlin: Callable,
                 obs_layer: [nn.Module, Policy] = None,
                 conv_padding_mode: str = 'circular',
                 tau_init: float = 1.,
                 tau_learnable: bool = True,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param dt: time step size
        :param hidden_size: number of neurons with potential
        :param conv_out_channels: number of filter for the 1-dim convolution along the potential-based neurons
        :param conv_kernel_size: size of the kernel for the 1-dim convolution along the potential-based neurons
        :param activation_nonlin: nonlinearity to compute the activations from the potential levels
        :param obs_layer: specify a custom PyTorch Module;
                          by default (`None`) a linear layer with biases is used
        :param tau_init: initial value for the shared time constant of the potentials
        :param tau_learnable: flag to determine if the time constant is a learnable parameter or a fixed tensor
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(dt, (float, int)):
            raise pyrado.TypeErr(given=dt, expected_type=float)
        if not isinstance(hidden_size, int):
            raise pyrado.TypeErr(given=hidden_size, expected_type=int)
        if hidden_size < 2:
            raise pyrado.ValueErr(given=hidden_size, g_constraint='15735c')
        if not conv_kernel_size%2 == 1:
            print_cbt(f'Made kernel size {conv_kernel_size} odd {conv_kernel_size + 1} for automated padding.', 'y')
            conv_kernel_size = conv_kernel_size + 1
        if not conv_padding_mode in ['circular', 'reflected', 'zeros']:
            raise pyrado.ValueErr(given=conv_padding_mode, eq_constraint='circular, reflected, or zeros')

        super().__init__(spec, use_cuda)

        # Store inputs
        self._dt = to.tensor([dt], dtype=to.get_default_dtype())
        self._input_size = spec.obs_space.flat_dim  # observations include goal distance, prediction error, ect.
        self._hidden_size = hidden_size  # number of potential neurons
        self._num_recurrent_layers = 1
        if not callable(activation_nonlin):
            if activation_nonlin is not None and not len(activation_nonlin) == spec.act_space.flat_dim:
                raise pyrado.ShapeErr(given=activation_nonlin, expected_match=spec.act_space.shape)
        self._activation_nonlin = activation_nonlin

        # Create the RNN's layers
        self.obs_layer = nn.Linear(self._input_size, self._hidden_size, bias=True) if obs_layer is None else obs_layer
        self.conv_layer = nn.Conv1d(
            in_channels=1,  # treat potentials as a time series of values (convolutions is over the "time" axis)
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size, stride=1,
            padding=conv_kernel_size//2 if conv_padding_mode != 'circular' else (conv_kernel_size + 1)//2,
            padding_mode=conv_padding_mode, dilation=1, groups=1, bias=False
        )
        # self.post_conv_layer = nn.Linear(conv_out_channels, spec.act_space.flat_dim, bias=False)
        self.act_layer = nn.Linear(self._hidden_size, spec.act_space.flat_dim, bias=False)

        # Call custom initialization function after PyTorch network parameter initialization
        self._potentials = to.zeros(self._hidden_size)
        self._init_potentials = to.zeros_like(self._potentials)
        self._potentials_max = 100.  # clip potentials symmetrically
        self._stimuli_internal = to.zeros_like(self._potentials)
        self._stimuli_external = to.zeros_like(self._potentials)

        # Potential dynamics's time constant
        self._tau_learnable = tau_learnable
        self._log_tau_init = to.log(to.tensor([tau_init], dtype=to.get_default_dtype()))
        self._log_tau = nn.Parameter(self._log_tau_init,
                                     requires_grad=True) if self._tau_learnable else self._log_tau_init

        # Initialize policy parameters
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    @property
    def hidden_size(self) -> int:
        """ Get the total number of hidden parameters is the hidden layer size times the hidden layer count. """
        assert self._num_recurrent_layers == 1
        return self._num_recurrent_layers*self._hidden_size

    @property
    def potentials(self) -> to.Tensor:
        """ Get the neurons' potentials. """
        return self._potentials

    @property
    def stimuli_external(self) -> to.Tensor:
        """
        Get the neurons' external stimuli, resulting from the current observations.
        This is used for recording during a rollout.
        """
        return self._stimuli_external

    @property
    def stimuli_internal(self) -> to.Tensor:
        """
        Get the neurons' internal stimuli, resulting from the previous activations of the neurons.
        This is used for recording during a rollout.
        """
        return self._stimuli_internal

    @property
    def tau(self) -> to.Tensor:
        """ Get the time scale parameter (exists for all potential dynamics functions). """
        return to.exp(self._log_tau)

    def potentials_dot(self, stimuli: to.Tensor) -> to.Tensor:
        """
        Compute the derivative of the neurons' potentials per time step.

        :param stimuli: sum of external stimuli at the current point in time
        :return: time derivative of the potentials
        """
        if not all(self.tau > 0):
            raise pyrado.ValueErr(given=self.tau, g_constraint='0')
        return (stimuli - self._potentials)/self.tau

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Initialize RNN layers
            init_param(self.obs_layer, **kwargs)
            init_param(self.conv_layer, **kwargs)
            # init_param(self.post_conv_layer, **kwargs)
            init_param(self.act_layer, **kwargs)

            # Initialize time constant if modifiable
            if self._tau_learnable:
                self._log_tau.data = self._log_tau_init

        else:
            self.param_values = init_values

    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        """
        Compute the goal distance, prediction error, and predicted cost.
        Then pass it to the wrapped RNN.

        :param obs: observations coming from the environment i.e. noisy
        :param hidden: current hidden states, in this case action and potentials of the last time step
        :return: current action and new hidden states
        """
        obs = obs.to(self.device)

        # We assume flattened observations, if they are 2d, they're batched.
        if len(obs.shape) == 1:
            batch_size = None
        elif len(obs.shape) == 2:
            batch_size = obs.shape[0]
        else:
            raise pyrado.ShapeErr(msg=f"Improper shape of 'obs'. Policy received {obs.shape},"
                                      f"but shape should be 1- or 2-dim")

        # Unpack hidden tensor (i.e. the potentials of the last step) if specified
        # The network can handle getting None by using default values
        potentials = self._unpack_hidden(hidden, batch_size) if hidden is not None else hidden

        # Don't track the gradient through the potentials
        potentials = potentials.detach()
        self._potentials = potentials

        # Clip the potentials, and save them for later use
        potentials = potentials.clamp(min=-self._potentials_max, max=self._potentials_max)

        # ----------------
        # Activation Logic
        # ----------------

        # Combine the current inputs
        self._stimuli_external = self.obs_layer(obs)

        # Pass the previous potentials through a nonlinearity
        potentials_conv = self._activation_nonlin(potentials)

        # Reshape and convolve
        b = batch_size if batch_size is not None else 1
        self._stimuli_internal = self.conv_layer(potentials_conv.view(b, 1, self._hidden_size))
        self._stimuli_internal = to.sum(self._stimuli_internal, dim=1)  # TODO do multiple out channels makes sense if just summed up?
        self._stimuli_internal = self._stimuli_internal.squeeze()

        # Combine the different output channels of the convolution
        # stimulus_pot = self.post_conv_layer(stimulus_pot)

        assert self._stimuli_external.shape == self._stimuli_internal.shape

        # Potential dynamics forward integration
        potentials = potentials + self._dt*self.potentials_dot(self._stimuli_external + self._stimuli_internal)

        # Compute the actions from the potentials
        act = self._activation_nonlin(potentials)
        act = self.act_layer(act)

        # # Since we want that this kind of Policy only returns activations in [0, 1] or in [-1, 1],
        # # we clip the actions right here
        # if self._output_nonlin is not to.sigmoid or self._output_nonlin is not to.tanh:
        #     act = act.clamp(min=-1., max=1.)

        # Pack hidden state
        hidden_out = self._pack_hidden(potentials, batch_size)

        # Return the next action and store the current potentials as a hidden variable
        return act, hidden_out

    def _unpack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Unpack the flat hidden state vector into a form the actual network module can use.
        Since hidden usually comes from some outer source, this method should validate it's shape.

        :param hidden: flat hidden state
        :param batch_size: if not `None`, hidden is 2-dim and the first dim represents parts of a data batch
        :return: unpacked hidden state of shape batch_size x channels_in x length_in, ready for the `Conv1d` module
        """
        if len(hidden.shape) == 1:
            assert hidden.shape[0] == self._num_recurrent_layers*self._hidden_size, \
                "Passed hidden variable's size doesn't match the one required by the network."
            assert batch_size is None, 'Cannot use batched observations with unbatched hidden state'
            return hidden.view(self._num_recurrent_layers*self._hidden_size)

        elif len(hidden.shape) == 2:
            assert hidden.shape[1] == self._num_recurrent_layers*self._hidden_size, \
                "Passed hidden variable's size doesn't match the one required by the network."
            assert hidden.shape[0] == batch_size, \
                f'Batch size of hidden state ({hidden.shape[0]}) must match batch size of observations ({batch_size})'
            return hidden.view(batch_size, self._num_recurrent_layers*self._hidden_size)

        else:
            raise RuntimeError(f"Improper shape of 'hidden'. Policy received {hidden.shape}, "
                               f"but shape should be 1- or 2-dim")

    def _pack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Pack the hidden state returned by the network into an 1-dim state vector.
        This should be the reverse operation of `_unpack_hidden`.

        :param hidden: hidden state as returned by the network
        :param batch_size: if not `None`, the result should be 2-dim and the first dim represents parts of a data batch
        :return: packed hidden state
        """
        if batch_size is None:
            # Simply flatten the hidden state
            return hidden.view(self._num_recurrent_layers*self._hidden_size)
        else:
            # Make sure that the batch dimension is the first element
            return hidden.view(batch_size, self._num_recurrent_layers*self._hidden_size)
