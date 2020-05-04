import functools
import sys
import torch as to
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from typing import Callable
from warnings import warn

from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_params import NormalParamNoise, SymmParamExplStrat
from pyrado.policies.linear import LinearPolicy
from pyrado.utils.optimizers import GSS
from pyrado.policies.base import Policy
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult


class REPS(ParameterExploring):
    """
    Episodic variant of Relative Entropy Policy Search (REPS)

    .. seealso::
        [1] J. Peters, K. Mülling, Y. Altuen, "Relative Entropy Policy Search", AAAI, 2010

        [2] This implementation was inspired by https://github.com/hanyas/rl/blob/master/rl/ereps/ereps.py
    """

    name: str = 'reps'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 max_iter: int,
                 eps: float,
                 gamma: float,
                 num_rollouts: int,
                 pop_size: int,
                 expl_std_init: float,
                 expl_std_min: float = 0.01,
                 symm_sampling: bool = False,
                 num_sampler_envs: int = 4,
                 num_epoch_dual: int = 1000,
                 use_map: bool = False,
                 grad_free_optim: bool = False,
                 lr_dual: float = 5e-4,
                 base_seed: int = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param eps: bound on the KL divergence between policy updates, e.g. 0.1
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param gamma: temporal discount factor; equal to 1 - reset probability
        :param pop_size: number of solutions in the population
        :param num_rollouts: number of rollouts per per policy sample
        :param expl_std_init: initial standard deviation for the exploration strategy
        :param expl_std_min: minimal standard deviation for the exploration strategy
        :param symm_sampling: use an exploration strategy which samples symmetric populations
        :param num_epoch_dual: number of epochs for the minimization of the dual function
        :param use_map: use maximum a-posteriori likelihood (`True`) or maximum likelihood (`False`) update rule
        :param grad_free_optim: use a derivative free optimizer (e.g. golden section search) or a SGD-based optimizer
        :param lr_dual: learning rate for the dual's optimizer (ignored if `grad_free_optim = True`)
        :param base_seed: seed added to all other seeds in order to make the experiments distinct but repeatable
        """
        if not isinstance(policy, LinearPolicy):
            warn('REPS is designed for linear policies only!', UserWarning)

        # Call ParameterExploring's constructor
        super().__init__(
            save_dir,
            env,
            policy,
            max_iter,
            num_rollouts,
            pop_size=pop_size,
            base_seed=base_seed,
            num_sampler_envs=num_sampler_envs,
        )

        # Store the inputs
        self.eps = eps
        self.gamma = gamma
        self.base_seed = base_seed
        self.use_map = use_map

        # Explore using normal noise
        self._expl_strat = NormalParamNoise(
            self._policy.num_param,
            full_cov=True,
            std_init=expl_std_init,
            std_min=expl_std_min,
        )
        if symm_sampling:
            # Exploration strategy based on symmetrical normally distributed noise
            # Symmetric buffer needs to have an even number of samples
            if self.pop_size % 2 != 0:
                self.pop_size += 1
            self._expl_strat = SymmParamExplStrat(self._expl_strat)

        self.kappa = to.tensor([0.], requires_grad=True)  # eta = exp(kappa)
        self._exp_min = -700.
        self._exp_max = 700.

        # Dual specific
        if grad_free_optim:
            self.optim_dual = GSS([{'params': self.kappa}],
                                  param_min=to.log(to.tensor([1e-4])),
                                  param_max=to.log(to.tensor([1e4])))
        else:
            self.optim_dual = to.optim.Adam([{'params': self.kappa}], lr=lr_dual, eps=1e-5)
            # self.optim_dual = to.optim.SGD([{'params': self.kappa}], lr=lr_dual, momentum=0.7, weight_decay=1e-4)
        self.num_epoch_dual = num_epoch_dual

    @property
    def eta(self) -> to.Tensor:
        r""" Get $\eta = e^{\kappa}$. """
        return to.exp(self.kappa)

    def weights(self, rets: to.Tensor) -> to.Tensor:
        """
        Compute the wights which are used to weights thy policy samples by their return

        :param rets: return values per policy sample after averaging over multiple rollouts using the same policy
        """
        shifted_rets = rets - to.max(rets)
        return to.exp(to.clamp(shifted_rets / self.eta, self._exp_min, self._exp_max))

    def dual(self, rets: to.Tensor) -> to.Tensor:
        """
        Compute the REPS dual function value.

        :param: dual loss value
        """
        w = self.weights(rets)
        return self.eta * self.eps + to.max(rets) + self.eta * to.log(to.mean(w))

    def policy_dual(self, param_samples: to.Tensor, w: to.Tensor) -> to.Tensor:
        """
        Compute the REPS policy-dual function value.

        :param param_samples:
        :param w: sample weights
        :return: dual loss value
        """
        distr_old = MultivariateNormal(self._policy.param_values, self._expl_strat.cov)
        self.wml(param_samples, w, eta=self.eta)

        distr_new = MultivariateNormal(self._policy.param_values, self._expl_strat.cov)
        logprobs = distr_new.log_prob(param_samples)
        kl_e = kl_divergence(distr_new, distr_old)  # mode seeking a.k.a. exclusive KL

        return w @ logprobs + self.eta * (self.eps - kl_e)

    def minimize(self,
                 loss_fcn: Callable,
                 rets: to.Tensor = None,
                 param_samples: to.Tensor = None,
                 w: to.Tensor = None):
        """
        Minimize the given dual function. Iterate num_epoch_dual times.

        :param loss_fcn: function to minimize
        :param rets: return values per policy sample after averaging over multiple rollouts using the same policy
        :param param_samples: all sampled policy parameters
        :param w: sample weights
        """
        if isinstance(self.optim_dual, GSS):
            self.optim_dual.reset()

        for _ in tqdm(range(self.num_epoch_dual), total=self.num_epoch_dual, desc=f'Minimizing dual', unit='epochs',
                      file=sys.stdout, leave=False):
            if not isinstance(self.optim_dual, GSS):
                # Reset the gradients
                self.optim_dual.zero_grad()

            # Compute value function loss
            if rets is not None and param_samples is None and w is None:
                loss = loss_fcn(rets)  # dual
            elif rets is None and param_samples is not None and w is not None:
                loss = loss_fcn(param_samples, w)  # policy dual
            else:
                raise NotImplementedError

            # Update the parameter
            if isinstance(self.optim_dual, GSS):
                if rets is not None and param_samples is None and w is None:
                    self.optim_dual.step(closure=functools.partial(loss_fcn, rets=rets))
                elif rets is None and param_samples is not None and w is not None:
                    self.optim_dual.step(closure=functools.partial(loss_fcn, param_samples=param_samples, w=w))
                else:
                    raise NotImplementedError

            else:
                loss.backward()
                self.optim_dual.step()

        if to.isnan(self.kappa):
            raise RuntimeError(f"The dual's optimization parameter kappa became NaN!")

    def wml(self, param_samples: to.Tensor, w: to.Tensor, eta: to.Tensor = to.tensor([0.])):
        """
        Weighted maximum likelihood update of the policy's mean and the exploration strategy's covariance

        :param param_samples: all sampled policy parameters
        :param w: sample weights
        :param eta: dual parameters
        """
        mean_old = self._policy.param_values.clone()
        cov_old = self._expl_strat.cov.clone()

        # Update the mean
        self._policy.param_values = (eta * mean_old + to.sum(w.view(-1, 1) * param_samples, dim=0)) / (to.sum(w) + eta)
        param_values_delta = self._policy.param_values - mean_old

        # Difference between all sampled policy parameters and the updated policy
        diff = param_samples - self._policy.param_values
        w_diff = to.einsum('nk,n,nh->kh', diff, w, diff)  # outer product of scaled diff, then sum over all samples

        # Update the covariance
        cov_new = (w_diff + eta * cov_old + eta * to.einsum('k,h->kh', param_values_delta, param_values_delta)
                   ) / (to.sum(w) + eta)
        self._expl_strat.adapt(cov=cov_new)

    def wmap(self, param_samples: to.Tensor, w: to.Tensor):
        """
        Weighted maximum a-posteriori likelihood update of the policy's mean and the exploration strategy's covariance

        :param param_samples: all sampled policy parameters
        :param w: sample weights
        """
        # Optimize for eta
        self.minimize(self.policy_dual, param_samples=param_samples, w=w.detach())
        # Update policy parameters
        self.wml(param_samples, w.detach(), eta=self.eta)

    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float = None):
        # Average the return values over the rollouts
        rets_avg_ros = param_results.mean_returns
        rets_avg_ros = to.from_numpy(rets_avg_ros)

        # Reset dual's parameter
        self.kappa.data.fill_(0.)

        # Dual
        with to.no_grad():
            distr_old = MultivariateNormal(self._policy.param_values, self._expl_strat.cov)
            loss = self.dual(rets_avg_ros)
            self.logger.add_value('dual loss before', loss.item())

        self.minimize(self.dual, rets=rets_avg_ros)

        with to.no_grad():
            loss = self.dual(rets_avg_ros)
            self.logger.add_value('dual loss after', loss.item())
            self.logger.add_value('eta', self.eta.item())

        # Compute the weights using the optimized eta
        w = self.weights(rets_avg_ros)

        # Update the policy's mean and the exploration strategy's covariance
        if self.use_map:
            self.wml(param_results.parameters, w)
        else:
            self.wmap(param_results.parameters, w)

        # Logging
        distr_new = MultivariateNormal(self._policy.param_values, self._expl_strat.cov)
        kl_e = kl_divergence(distr_new, distr_old)  # mode seeking a.k.a. exclusive KL
        kl_i = kl_divergence(distr_old, distr_new)  # mean seeking a.k.a. inclusive KL
        self.logger.add_value('min expl strat std', to.min(self._expl_strat.std))
        self.logger.add_value('avg expl strat std', to.mean(self._expl_strat.std.data).detach().numpy())
        self.logger.add_value('max expl strat std', to.max(self._expl_strat.std))
        self.logger.add_value('expl strat entropy', self._expl_strat.get_entropy().item())
        self.logger.add_value('KL(new_old)', kl_e.item())
        self.logger.add_value('KL(old_new)', kl_i.item())
