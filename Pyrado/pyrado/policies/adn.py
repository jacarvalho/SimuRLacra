import torch as to
import torch.nn as nn
from typing import Callable, Sequence

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy, PositiveScaleLayer
from pyrado.policies.base_recurrent import RecurrentPolicy
from pyrado.policies.initialization import init_param


def pd_linear(p: to.Tensor, s: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Basic proportional dynamics $\tau \dot{p} = -(s+p)$

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint='0')
    return -(s + p)/tau


def pd_cubic(p: to.Tensor, s: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Basic proportional dynamics with additional cubic decay $\tau \dot{p} = -(s+p) - \kappa p^3$

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint='0')
    if not all(kwargs['kappa'] >= 0):
        raise pyrado.ValueErr(given=kwargs['kappa'], ge_constraint='0')
    return (-(s + p) - kwargs['kappa']*to.pow((s + p), 3))/tau


def pd_capacity_21(p: to.Tensor, s: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Capacity-based dynamics with 2 stable ($p=-C$, $p=C$) and 1 unstable fix points ($p=0$) for $s=0$

    .. note::
        Intended to be used with sigmoid activation function, e.g. for the position tasks in RcsPySim.

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint='0')
    return ((s + p)*(to.ones_like(p) - (s + p) ** 2/kwargs['capacity'] ** 2))/tau


def pd_capacity_21_abs(p: to.Tensor, s: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Capacity-based dynamics with 2 stable ($p=-C$, $p=C$) and 1 unstable fix points ($p=0$) for $s=0$
    The "absolute version" has a lower magnitude and a lower oder of the resulting polynomial.

    .. note::
        Intended to be used with sigmoid activation function, e.g. for the position tasks in RcsPySim.

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint='0')
    return ((s + p)*(to.ones_like(p) - to.abs((s + p))/kwargs['capacity']))/tau


def pd_capacity_32(p: to.Tensor, s: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Capacity-based dynamics with 3 stable ($p=-C$, $p=0$, $p=C$) and 2 unstable fix points ($p=-C/2$, $p=C/2$) for $s=0$

    .. note::
        Intended to be used with tanh activation function, e.g. for the velocity tasks in RcsPySim.

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint='0')
    return ((s + p)*(to.ones_like(p) - (s + p) ** 2/kwargs['capacity'] ** 2)*
            (to.ones_like(p) - (2*(s + p)) ** 2/kwargs['capacity'] ** 2))/-tau


def pd_capacity_32_abs(p: to.Tensor, s: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Capacity-based dynamics with 3 stable ($p=-C$, $p=0$, $p=C$) and 2 unstable fix points ($p=-C/2$, $p=C/2$) for $s=0$
    The "absolute version" is less skewed and symmetric due to a lower oder of the resulting polynomial.

    .. note::
        Intended to be used with tanh activation function, e.g. for the velocity tasks in RcsPySim.

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint='0')
    return ((s + p)*(to.ones_like(p) - to.abs((s + p))/kwargs['capacity'])*
            (to.ones_like(p) - 2*to.abs((s + p))/kwargs['capacity']))/-tau


class ADNPolicy(RecurrentPolicy):
    """
    Activation Dynamic Network (ADN)
    
    .. seealso::
        [1] T. Luksch, M. Gineger, M. Mühlig, T. Yoshiike, "Adaptive Movement Sequences and Predictive Decisions based
        on Hierarchical Dynamical Systems", IROS, 2012
    """

    def __init__(self,
                 spec: EnvSpec,
                 dt: float,
                 output_nonlin: [Callable, Sequence[Callable]],
                 potentials_dyn_fcn: Callable,
                 obs_layer: [nn.Module, Policy] = None,
                 tau_init: float = 2.,
                 tau_learnable: bool = True,
                 kappa_init: float = 0.01,
                 kappa_learnable: bool = True,
                 capacity_learnable: bool = True,
                 scaling_layer: bool = True,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param dt: time step size
        :param output_nonlin: nonlinearity for output layer, highly suggested functions:
                              `to.sigmoid` for position `to.tasks`, tanh for velocity tasks
        :param potentials_dyn_fcn: function to compute the derivative of the neurons' potentials
        :param obs_layer: specify a custom Pytorch Module;
                          by default (`None`) a linear layer with biases is used
        :param tau_init: initial value for the shared time constant of the potentials
        :param tau_learnable: flag to determine if the time constant is a learnable parameter or a fixed tensor
        :param kappa_init: initial value for the cubic decay
        :param kappa_learnable: flag to determine if cubic decay is a learnable parameter or a fixed tensor
        :param capacity_learnable: flag to determine if capacity is a learnable parameter or a fixed tensor
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
        self._hidden_size = spec.act_space.flat_dim  # hidden_size = output_size = num actions
        self._num_recurrent_layers = 1
        if not callable(output_nonlin):
            if output_nonlin is not None and not len(output_nonlin) == spec.act_space.flat_dim:
                raise pyrado.ShapeErr(given=output_nonlin, expected_match=spec.act_space.shape)
        self._output_nonlin = output_nonlin
        self._potentials_dot_fcn = potentials_dyn_fcn

        # Create the RNN's layers
        self.obs_layer = nn.Linear(self._input_size, self._hidden_size, bias=True) if obs_layer is None else obs_layer
        self.prev_act_layer = nn.Linear(self._hidden_size, self._hidden_size, bias=False)
        if scaling_layer:
            self.scaling_layer = PositiveScaleLayer(self._hidden_size)  # see beta in eq (4) of [1]
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
        # cubic decay
        if potentials_dyn_fcn == pd_cubic:
            self._kappa_learnable = kappa_learnable
            self._log_kappa_init = to.log(to.tensor([kappa_init], dtype=to.get_default_dtype()))
            self._log_kappa = nn.Parameter(self._log_kappa_init,
                                           requires_grad=True) if self._kappa_learnable else self._log_kappa_init
        else:
            self._log_kappa = None
        # capacity
        self._capacity_learnable = capacity_learnable
        if potentials_dyn_fcn in [pd_capacity_21, pd_capacity_21_abs]:
            self._log_capacity_init = to.log(to.tensor([5.], dtype=to.get_default_dtype()))
            self._log_capacity = nn.Parameter(self._log_capacity_init, requires_grad=True) \
                if self._capacity_learnable else self._log_capacity_init
        elif potentials_dyn_fcn in [pd_capacity_32, pd_capacity_32_abs]:
            self._log_capacity_init = to.log(to.tensor([3.], dtype=to.get_default_dtype()))
            self._log_capacity = nn.Parameter(self._log_capacity_init, requires_grad=True) \
                if self._capacity_learnable else self._log_capacity_init
        else:
            self._log_capacity = None

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

    @property
    def kappa(self) -> [None, to.Tensor]:
        """ Get the cubic decay parameter (exists for cubic decay-based dynamics functions), else return `None`. """
        if self._log_kappa is None:
            return None
        else:
            return to.exp(self._log_kappa)

    @property
    def capacity(self) -> [None, to.Tensor]:
        """ Get the time scale parameter (exists for capacity-based dynamics functions), else return `None`. """
        if self._log_capacity is None:
            return None
        else:
            return to.exp(self._log_capacity)

    def potentials_dot(self, stimuli: to.Tensor) -> to.Tensor:
        """
        Compute the derivative of the neurons' potentials per time step.

        :param stimuli: sum of external stimuli at the current point in time
        :return: time derivative of the potentials
        """
        return self._potentials_dot_fcn(self._potentials, stimuli, self.tau, kappa=self.kappa, capacity=self.capacity)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Initialize RNN layers
            init_param(self.obs_layer, **kwargs)
            init_param(self.prev_act_layer, **kwargs)
            if kwargs.get('sigmoid_nlin', False):
                self.prev_act_layer.weight.data.fill_(-0.5)  # inhibit others
                for i in range(self.prev_act_layer.weight.data.shape[0]):
                    self.prev_act_layer.weight.data[i, i] = 1.  # excite self
            init_param(self.scaling_layer, **kwargs)

            # Initialize time constant if modifiable
            if self._tau_learnable:
                self._log_tau.data = self._log_tau_init
            # Initialize cubic decay if modifiable
            if self._potentials_dot_fcn == pd_cubic:
                if self._kappa_learnable:
                    self._log_kappa.data = self._log_kappa_init
            # Initialize capacity if modifiable
            elif self._potentials_dot_fcn in [pd_capacity_21, pd_capacity_21_abs, pd_capacity_32, pd_capacity_32_abs]:
                if self._capacity_learnable:
                    self._log_capacity.data = self._log_capacity_init

        else:
            self.param_values = init_values

    def init_hidden(self, batch_size: int = None):
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
        :param hidden: current hidden states, in this case the linearly combined actions of the last step
        :return: action and new hidden states i.e. linearly combined actions
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
        # Don't track gradient through potentials
        potentials = potentials.detach()

        # Save potentials for later use in self.potentials_dot()
        self._potentials = potentials

        # ----------------
        # Activation Logic
        # ----------------

        # Combine the current input and the hidden variables from the last step
        stimulus_obs = self.obs_layer(obs)
        stimulus_prev_act = self.prev_act_layer(prev_act)
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

        # Clip the potentials too
        potentials = potentials.clamp(min=-self._potentials_max, max=self._potentials_max)

        # Save potentials for later use
        self._potentials = potentials

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
