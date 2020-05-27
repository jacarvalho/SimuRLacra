import torch as to
from warnings import warn

import pyrado
from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_params import NormalParamNoise, SymmParamExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.linear import LinearPolicy
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult


class PoWER(ParameterExploring):
    """
    Return-based variant of Policy learning by Weighting Exploration with the Returns (PoWER)

    .. note::
        PoWER was designed for linear policies.
        PoWER is must use positive reward functions (improper probability distribution) [1, p.10].
        The original implementation is tailored to movement primitives like DMPs.

    .. seealso::
        [1] J. Kober and J. Peters, "Policy Search for Motor Primitives in Robotics", Machine Learning, 2011
    """

    name: str = 'power'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 max_iter: int,
                 pop_size: int,
                 num_rollouts: int,
                 num_is_samples: int,
                 expl_std_init: float,
                 expl_std_min: float = 0.01,
                 symm_sampling: bool = False,
                 num_sampler_envs: int = 4,
                 base_seed: int = None,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param pop_size: number of solutions in the population
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_rollouts: number of rollouts per policy sample
        :param num_is_samples: of samples (policy parameter sets & returns) for importance sampling
        :param expl_std_init: initial standard deviation for the exploration strategy
        :param expl_std_min: minimal standard deviation for the exploration strategy
        :param symm_sampling: use an exploration strategy which samples symmetric populations
        :param base_seed: seed added to all other seeds in order to make the experiments distinct but repeatable
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(policy, LinearPolicy):
            warn('PoWER is designed for linear policies!', UserWarning)

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
            logger=logger,
        )

        # Explore using normal noise
        self._expl_strat = NormalParamNoise(
            self._policy.num_param,
            full_cov=True,
            std_init=expl_std_init,
            std_min=expl_std_min,
        )
        if symm_sampling:
            # Exploration strategy based on symmetrical normally distributed noise
            if self.pop_size%2 != 0:
                # Symmetric buffer needs to have an even number of samples
                self.pop_size += 1
            self._expl_strat = SymmParamExplStrat(self._expl_strat)

        # Initialize memory for importance sampling
        self.num_is_samples = min(pop_size, num_is_samples)
        self.is_mem_ret = 1e-6*to.ones(self.num_is_samples)  # has to be initialized > 0 due to first covariance update
        self.is_mem_params = to.zeros(self.num_is_samples, self._policy.num_param)
        self.is_mem_W = to.zeros(self.num_is_samples, self._policy.num_param, self._policy.num_param)

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Reset memory for importance sampling
        self.is_mem_ret = 1e-6*to.ones(self.num_is_samples)  # has to be initialized > 0 due to first covariance update
        self.is_mem_params = to.zeros(self.num_is_samples, self._policy.num_param)
        self.is_mem_W = to.zeros(self.num_is_samples, self._policy.num_param, self._policy.num_param)

    @to.no_grad()
    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float = None):
        # Average the return values over the rollouts
        rets_avg_ros = to.tensor(param_results.mean_returns)
        if any(rets_avg_ros < 0):
            warn('PoWER is must use positive reward functions (improper probability distribution)!')

        # We do the simplification from the original implementation, which is only valid for the return-based variant
        W = to.inverse(self._expl_strat.noise.cov)

        # For importance sampling we select the best rollouts
        self.is_mem_ret = to.cat([self.is_mem_ret, rets_avg_ros], dim=0)
        self.is_mem_params = to.cat([self.is_mem_params, param_results.parameters], dim=0)
        self.is_mem_W = to.cat([self.is_mem_W, W.repeat(self.pop_size, 1, 1)], dim=0)  # same cov for all rollouts

        # Descending sort according to return values
        idcs_dcs = to.argsort(self.is_mem_ret, descending=True)
        self.is_mem_ret = self.is_mem_ret[idcs_dcs]
        self.is_mem_params = self.is_mem_params[idcs_dcs, :]
        self.is_mem_W = self.is_mem_W[idcs_dcs, :, :]

        # Update the exploration covariance (see [1, p.32]). We use all rollouts to avoid rapid convergence to 0.
        eps = self.is_mem_params - self._policy.param_values  # policy parameter perturbations
        cov_num = to.einsum('nj,nk,n->jk', eps, eps, self.is_mem_ret)  # weighted outer product
        cov_dnom = sum(self.is_mem_ret)
        self._expl_strat.adapt(cov=cov_num/(cov_dnom + 1e-8))

        # Only memorize the best parameter sets & returns (importance sampling)
        self.is_mem_ret = self.is_mem_ret[:self.num_is_samples]
        self.is_mem_params = self.is_mem_params[:self.num_is_samples, :]
        self.is_mem_W = self.is_mem_W[:self.num_is_samples, :, :]

        # Update the policy mean (see [1, p.10])
        eps = eps[:self.num_is_samples, :]
        mean_num = to.einsum('njk,nj,n->k', self.is_mem_W, eps, self.is_mem_ret)  # weighted dot product
        mean_dnom = to.einsum('njk,n->jk', self.is_mem_W, self.is_mem_ret)  # weighted sum
        inv_dnom = to.inverse(mean_dnom + 1e-8)
        self._policy.param_values += to.matmul(inv_dnom, mean_num)

        # Logging
        self.logger.add_value('min expl strat std', to.min(self._expl_strat.std))
        self.logger.add_value('avg expl strat std', to.mean(self._expl_strat.std.data).detach().numpy())
        self.logger.add_value('max expl strat std', to.max(self._expl_strat.std))
        self.logger.add_value('expl strat entropy', self._expl_strat.get_entropy().item())
