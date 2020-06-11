import torch as to
import torch.nn as nn
from typing import Callable, Sequence

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy, PositiveScaleLayer
from pyrado.policies.base_recurrent import RecurrentPolicy
from pyrado.policies.initialization import init_param


class NFPolicy(RecurrentPolicy):
    """
    Neural Fields (NF)

    https://discuss.pytorch.org/t/understanding-convolution-1d-output-and-input/30764/3
    
    .. seealso::
        [1] S.-I. Amari "Dynamics of Pattern Formation in Lateral-Inhibition Type Neural Fields",
        Biological Cybernetics, 1977
    """

    def __init__(self,
                 spec: EnvSpec,
                 dt: float,
                 hidden_size: int,
                 output_nonlin: [Callable, Sequence[Callable]],
                 obs_layer: [nn.Module, Policy] = None,
                 tau_init: float = 2.,
                 tau_learnable: bool = True,
                 scaling_layer: bool = True,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param dt: time step size
        :param hidden_size:
        :param output_nonlin: nonlinearity for output layer, highly suggested functions:
                              `to.sigmoid` for position `to.tasks`, tanh for velocity tasks
        :param obs_layer: specify a custom Pytorch Module;
                          by default (`None`) a linear layer with biases is used
        :param tau_init: initial value for the shared time constant of the potentials
        :param tau_learnable: flag to determine if the time constant is a learnable parameter or a fixed tensor
        :param scaling_layer: add a scaling before the nonlinearity which converts the potentials to activations
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)
        if not isinstance(dt, (float, int)):
            raise pyrado.TypeErr(given=dt, expected_type=float)

        # Store inputs
        self._dt = to.tensor([dt], dtype=to.get_default_dtype())
        self._input_size = spec.obs_space.flat_dim  # observations include goal distance, prediction error, ect.
        self._hidden_size = hidden_size  # number of potential neurons
        assert self._hidden_size == spec.act_space.flat_dim, "Still, we have as many pot neurons as actions"
        self._num_recurrent_layers = 1
        if not callable(output_nonlin):
            if output_nonlin is not None and not len(output_nonlin) == spec.act_space.flat_dim:
                raise pyrado.ShapeErr(given=output_nonlin, expected_match=spec.act_space.shape)
        self._output_nonlin = output_nonlin

        # Create the RNN's layers
        self.obs_layer = nn.Linear(self._input_size, self._hidden_size, bias=True) if obs_layer is None else obs_layer
        self.prev_act_layer = nn.Conv1d(
            in_channels=spec.act_space.flat_dim, out_channels=self._hidden_size, kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=False, padding_mode='zeros'
        )
        if scaling_layer:
            self.scaling_layer = PositiveScaleLayer(spec.act_space.flat_dim)  # see beta in eq (4) of [1]
        else:
            self.scaling_layer = None

        # Call custom initialization function after PyTorch network parameter initialization
        self._potentials = to.zeros(self._hidden_size)
        self._init_potentials = to.zeros_like(self._potentials)
        self._potentials_max = 100.  # clip potentials symmetrically
        self._stimuli = to.zeros_like(self._potentials)

        # Potential dynamics
        # time constant
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
        return self._num_recurrent_layers*self._hidden_size + self._hidden_size  # added once for the potentials

    @property
    def potentials(self) -> to.Tensor:
        """ Get the neurons' potentials. """
        return self._potentials

    @property
    def stimuli(self) -> to.Tensor:
        """ Get the neurons' (external) stimuli. This is used for recording during a rollout """
        return self._stimuli

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
            init_param(self.prev_act_layer, **kwargs)
            init_param(self.scaling_layer, **kwargs)

            # Initialize time constant if modifiable
            if self._tau_learnable:
                self._log_tau.data = self._log_tau_init

        else:
            self.param_values = init_values

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        """
        Provide initial values for the hidden parameters. This should usually be a zero tensor.

        :param batch_size: number of states to track in parallel
        :return: Tensor of batch_size x hidden_size
        """
        return self._pack_hidden(*self._init_hidden_unpacked(batch_size), batch_size=batch_size)

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

        # Unpack hidden tensor if specified
        if hidden is not None:
            prev_act, potentials = self._unpack_hidden(hidden, batch_size)
        else:
            prev_act, potentials = self._init_hidden_unpacked(batch_size)

        # Don't track the gradient through the potentials
        potentials = potentials.detach()

        # Clip the potentials, and save them for later use
        potentials = potentials.clamp(min=-self._potentials_max, max=self._potentials_max)
        self._potentials = potentials

        # ----------------
        # Activation Logic
        # ----------------

        # Combine the current input and the hidden variables from the last step
        stimulus_obs = self.obs_layer(obs)
        stimulus_prev_act = self.prev_act_layer(prev_act)
        assert stimulus_obs.shape == stimulus_prev_act.shape
        self._stimuli = stimulus_obs + stimulus_prev_act

        # Potential dynamics forward integration
        potentials = potentials + self._dt*self.potentials_dot(self._stimuli)

        # Optionally scale the potentials
        act = self.scaling_layer(potentials) if self.scaling_layer is not None else potentials

        # Pass the potentials through a nonlinearity
        if self._output_nonlin is not None:
            if isinstance(self._output_nonlin, (list, tuple)):
                for i in range(len(act)):
                    # Individual nonlinearity for each dimension
                    act[i] = self._output_nonlin[i](act[i])
            else:
                # Same nonlinearity for all dimensions
                act = self._output_nonlin(act)

        # Since we want that this kind of Policy only returns activations in [0, 1] or in [-1, 1],
        # we clip the actions right here
        if self._output_nonlin is not to.sigmoid or self._output_nonlin is not to.tanh:
            act = act.clamp(min=-1., max=1.)

        # Pack hidden state
        prev_act = act.clone()
        hidden_out = self._pack_hidden(prev_act, potentials, batch_size)

        # Return the next action and store the last one as a hidden variable
        return act, hidden_out

    def _init_hidden_unpacked(self, batch_size: int = None):
        """ Get initial hidden variables in unpacked state """
        assert self._num_recurrent_layers == 1
        # Obtain values
        hidden = to.zeros(self._num_recurrent_layers*self._hidden_size)
        potentials = self._init_potentials.clone()

        # Batch if needed
        if batch_size is not None:
            hidden = hidden.unsqueeze(0).expand(batch_size, -1)
            potentials = potentials.unsqueeze(0).expand(batch_size, -1)

        return hidden, potentials

    def _unpack_hidden(self, packed: to.Tensor, batch_size: int = None):
        """ Unpack hidden values from argument """
        n_rh = self._num_recurrent_layers*self._hidden_size
        # Split into hidden and potentials
        return packed[..., :n_rh], packed[..., n_rh:]

    def _pack_hidden(self, prev_act: to.Tensor, potentials: to.Tensor, batch_size: int = None):
        """ Pack hidden values """
        return to.cat([prev_act, potentials], dim=-1)
