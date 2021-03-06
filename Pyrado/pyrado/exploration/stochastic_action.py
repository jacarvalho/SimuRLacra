import torch as to
import torch.nn as nn
from abc import ABC
from torch.distributions import Distribution, Bernoulli, Categorical

import pyrado
from pyrado.policies.base import Policy
from pyrado.exploration.normal_noise import DiagNormalNoise
from pyrado.exploration.uniform_noise import UniformNoise
from pyrado.policies.two_headed import TwoHeadedPolicy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.math import clamp
from pyrado.utils.properties import Delegate
from pyrado.utils.tensor import atleast_2D


class StochasticActionExplStrat(Policy, ABC):
    """ Explore by sampling actions from a distribution. """

    def __init__(self, policy: Policy):
        """
        Constructor

        :param policy: wrapped policy
        """
        super().__init__(policy.env_spec)
        self.policy = policy

    @property
    def is_recurrent(self) -> bool:
        return self.policy.is_recurrent

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        return self.policy.init_hidden(batch_size)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        self.policy.init_param(init_values, **kwargs)

    def reset(self):
        self.policy.reset()

    def forward(self, obs: to.Tensor, *extra) -> (to.Tensor, tuple):
        # Get actions from policy
        if self.policy.is_recurrent:
            if isinstance(self.policy, TwoHeadedPolicy):
                act, other, hidden = self.policy(obs, *extra)
            else:
                act, hidden = self.policy(obs, *extra)
        else:
            if isinstance(self.policy, TwoHeadedPolicy):
                act, other = self.policy(obs, *extra)
            else:
                act = self.policy(obs, *extra)

        # Compute exploration (use rsample to apply the reparametrization trick  if needed)
        act_expl = self.action_dist_at(act).rsample()  # act is the mean if train_mean=False

        # Return the exploratove actions and optionally the other policy outputs
        if self.policy.is_recurrent:
            if isinstance(self.policy, TwoHeadedPolicy):
                return act_expl, other, hidden
            else:
                return act_expl, hidden
        else:
            if isinstance(self.policy, TwoHeadedPolicy):
                return act_expl, other
            else:
                return act_expl

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = 'hidden_states') -> Distribution:
        """
        Re-evaluate the given rollout using the policy wrapped by this exploration strategy.
        Use this method to get gradient data on the action distribution.

        :param rollout: recorded, complete rollout
        :param hidden_states_name: name of hidden states rollout entry, used for recurrent networks.
        :return: actions with gradient data
        """
        self.policy.eval()
        if isinstance(self.policy, TwoHeadedPolicy):
            acts, _ = self.policy.evaluate(rollout, hidden_states_name)  # ignore the second head's output
        else:
            acts = self.policy.evaluate(rollout, hidden_states_name)
        return self.action_dist_at(acts)

    def action_dist_at(self, policy_output: to.Tensor) -> Distribution:
        """
        Return the action distribution for the given output from the wrapped policy.

        :param policy_output: output from the wrapped policy, i.e. the noise-free action values
        :return: action distribution
        """
        raise NotImplementedError


class NormalActNoiseExplStrat(StochasticActionExplStrat):
    """ Exploration strategy which adds Gaussian noise to the continuous policy actions """

    def __init__(self,
                 policy: Policy,
                 std_init: [float, to.Tensor],
                 std_min: [float, to.Tensor] = 0.01,
                 train_mean: bool = False,
                 learnable: bool = True):
        """
        Constructor

        :param policy: wrapped policy
        :param std_init: initial standard deviation for the exploration noise
        :param std_min: minimal standard deviation for the exploration noise
        :param train_mean: set `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param learnable: `True` if the parameters should be tuneable (default), `False` for shallow use (just sampling)
        """
        super().__init__(policy)
        self._noise = DiagNormalNoise(
            noise_dim=policy.env_spec.act_space.flat_dim,
            std_init=std_init,
            std_min=std_min,
            train_mean=train_mean,
            learnable=learnable
        )

    @property
    def noise(self) -> DiagNormalNoise:
        """ Get the exploration noise. """
        return self._noise

    def action_dist_at(self, policy_output: to.Tensor) -> Distribution:
        return self._noise(policy_output)

    # Make NormalActNoiseExplStrat appear as if it would have the following functions / properties
    reset_expl_params = Delegate('_noise')
    std = Delegate('_noise')
    mean = Delegate('_noise')
    get_entropy = Delegate('_noise')


class UniformActNoiseExplStrat(StochasticActionExplStrat):
    """ Exploration strategy which adds uniform noise to the continuous policy actions """

    def __init__(self,
                 policy: Policy,
                 halfspan_init: [float, to.Tensor],
                 halfspan_min: [float, list] = 0.01,
                 train_mean: bool = False,
                 learnable: bool = True):
        """
        Constructor

        :param policy: wrapped policy
        :param halfspan_init: initial value of the half interval for the exploration noise
        :param halfspan_min: minimal standard deviation for the exploration noise
        :param train_mean: set `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param learnable: `True` if the parameters should be tuneable (default), `False` for shallow use (just sampling)
        """
        super().__init__(policy)
        self._noise = UniformNoise(
            noise_dim=policy.env_spec.act_space.flat_dim,
            halfspan_init=halfspan_init,
            halfspan_min=halfspan_min,
            train_mean=train_mean,
            learnable=learnable
        )

    @property
    def noise(self) -> UniformNoise:
        """ Get the exploration noise. """
        return self._noise

    def action_dist_at(self, policy_output: to.Tensor) -> Distribution:
        return self._noise(policy_output)

    # Make NormalActNoiseExplStrat appear as if it would have the following functions / properties
    reset_expl_params = Delegate('_noise')
    halfspan = Delegate('_noise')
    get_entropy = Delegate('_noise')


class SACExplStrat(StochasticActionExplStrat):
    """
    State-dependent exploration strategy which adds normal noise squashed into by a tanh to the continuous actions.

    .. note::
        This exploration strategy is specifically designed for SAC.
        Due to the tanh transformation, it returns action values within [-1,1].
    """

    def __init__(self, policy: Policy, std_init: [float, to.Tensor]):
        """
        Constructor

        :param policy: wrapped policy
        :param std_init: initial standard deviation for the exploration noise
        """
        if not isinstance(policy, TwoHeadedPolicy):
            raise pyrado.TypeErr(given=policy, expected_type=TwoHeadedPolicy)
        super().__init__(policy)

        # Do not need to learn the exploration noise via an optimizer, since it is handled by the policy in this case
        self._noise = DiagNormalNoise(
            noise_dim=policy.env_spec.act_space.flat_dim,
            std_init=std_init,
            std_min=0.,  # ignore since we are explicitly clipping in log space later
            train_mean=False,
            learnable=False
        )

        self._log_std_min = to.tensor(-10.)
        self._log_std_max = to.tensor(1.)

    @property
    def noise(self) -> DiagNormalNoise:
        """ Get the exploration noise. """
        return self._noise

    def action_dist_at(self, policy_output_1: to.Tensor, policy_output_2: to.Tensor) -> Distribution:
        """
        Return the action distribution for the given output from the wrapped policy.
        This method is made for two-headed policies, e.g. used with SAC.

        :param policy_output_1: first head's output from the wrapped policy, noise-free action values
        :param policy_output_2: first head's output from the wrapped policy, state-dependent log std values
        :return: action distribution
        """
        # Manually adapt the Gaussian's covariance
        self._noise.std = to.exp(policy_output_2)
        return self._noise(policy_output_1)

    # Make NormalActNoiseExplStrat appear as if it would have the following functions / properties
    reset_expl_params = Delegate('_noise')
    std = Delegate('_noise')
    mean = Delegate('_noise')
    get_entropy = Delegate('_noise')

    def forward(self, obs: to.Tensor, *extra) -> [(to.Tensor, to.Tensor), (to.Tensor, to.Tensor, to.Tensor)]:
        # Get actions from policy (which for this class always have a two-headed architecture)
        if self.policy.is_recurrent:
            act, log_std, hidden = self.policy(obs, *extra)
        else:
            act, log_std = self.policy(obs, *extra)

        # Clamp the log_std coming from the policy
        log_std = clamp(log_std, lo=self._log_std_min, up=self._log_std_max)

        # Compute exploration (use rsample to apply the reparametrization trick)
        noise = self.action_dist_at(act, log_std)
        u = noise.rsample()
        act_expl = to.tanh(u)
        log_prob = noise.log_prob(u)
        log_prob = self._enforce_act_expl_bounds(log_prob, act_expl)

        if self.policy.is_recurrent:
            return act_expl, log_std, hidden
        else:
            return act_expl, log_prob

    @staticmethod
    def _enforce_act_expl_bounds(log_probs: to.Tensor, act_expl: to.Tensor, eps: float = 1e-6):
        r"""
        Transform the `log_probs` accounting for the squashed tanh exploration

        .. seealso::
            Eq. (21) in [2]

        :param log_probs: $\log( \mu(u|s) )$
        :param act_expl: action values with explorative noise
        :param eps: additive term for numerical stability of the log
        :return: $\log( \pi(a|s) )$
        """
        # Batch dim along the first dim
        act_expl_ = atleast_2D(act_expl)
        log_probs_ = atleast_2D(log_probs)

        # Sum over action dimensions
        log_probs_ = to.sum(log_probs_ - to.log(to.ones_like(act_expl_) - to.pow(act_expl_, 2) + eps), 1, keepdim=True)
        if act_expl_.shape[0] > 1:
            return log_probs_  # batched mode
        else:
            return log_probs_.squeeze(1)  # one sample at a time

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = 'hidden_states') -> Distribution:
        """
        Re-evaluate the given rollout using the policy wrapped by this exploration strategy.
        Use this method to get gradient data on the action distribution.
        This version is tailored to the two-headed policy architecture used for SAC, since it requires a two-headed
        policy, where the first head returns the mean action and the second head returns the state-dependent std.

        :param rollout: recorded, complete rollout
        :param hidden_states_name: name of hidden states rollout entry, used for recurrent networks.
        :return: actions with gradient data
        """
        self.policy.eval()
        acts, log_stds = self.policy.evaluate(rollout, hidden_states_name)
        return self.action_dist_at(acts, log_stds)


class EpsGreedyExplStrat(StochasticActionExplStrat):
    """ Exploration strategy which selects discrete actions epsilon-greedily """

    def __init__(self, policy: Policy, eps: float = 1., eps_schedule_gamma: float = 0.99, eps_final: float = 0.05):
        """
        Constructor

        :param policy: wrapped policy
        :param eps: parameter determining the greediness, can be optimized or scheduled
        :param eps_schedule_gamma: temporal discount factor for the exponential decay of epsilon
        :param eps_final: minimum value of epsilon
        """
        super().__init__(policy)
        self.eps = nn.Parameter(to.tensor(eps), requires_grad=True)
        self._eps_init = to.tensor(eps)
        self._eps_final = to.tensor(eps_final)
        self._eps_old = to.tensor(eps)
        self.eps_gamma = eps_schedule_gamma
        self.distr_eps = Bernoulli(probs=self.eps.data)  # eps chance to sample 1

        flat_dim = self.policy.env_spec.act_space.flat_dim
        self.distr_act = Categorical(to.ones(flat_dim)/flat_dim)

    def eval(self):
        """ Call PyTorch's eval function and set the deny every exploration. """
        super(Policy, self).eval()
        self._eps_old = self.eps.clone()
        self.eps.data = to.tensor(0.)
        self.distr_eps = Bernoulli(probs=self.eps.data)

    def train(self, mode=True):
        """ Call PyTorch's eval function and set the re-activate every exploration. """
        super(Policy, self).train()
        self.eps = nn.Parameter(self._eps_old, requires_grad=True)
        self.distr_eps = Bernoulli(probs=self.eps.data)

    def schedule_eps(self, steps: int):
        self.eps.data = self._eps_final + (self._eps_init - self._eps_final)*self.eps_gamma ** steps
        self.distr_eps = Bernoulli(probs=self.eps.data)

    def forward(self, obs: to.Tensor, *extra) -> (to.Tensor, tuple):
        # Get exploiting action from policy given the current observation (this way we always get a value for hidden)
        if self.policy.is_recurrent:
            act, hidden = self.policy(obs, *extra)
        else:
            act = self.policy(obs, *extra)

        # Compute epsilon-greedy exploration
        if self.distr_eps.sample() == 1:
            act_idx = self.distr_act.sample()
            act = self.env_spec.act_space.eles[int(act_idx)]
            act = to.from_numpy(act).to(to.get_default_dtype())

        if self.policy.is_recurrent:
            return act, hidden
        else:
            return act

    def action_dist_at(self, policy_output: to.Tensor) -> Distribution:
        # Not needed for this exploration strategy
        raise NotImplementedError
