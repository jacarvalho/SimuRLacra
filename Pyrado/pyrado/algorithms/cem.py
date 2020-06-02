import os.path as osp
import torch as to
from copy import deepcopy

import pyrado
from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.environments.base import Env
from pyrado.exploration.normal_noise import FullNormalNoise, DiagNormalNoise
from pyrado.exploration.stochastic_params import NormalParamNoise, SymmParamExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult
from pyrado.utils.math import cov


class CEM(ParameterExploring):
    r"""
    Cross-Entropy Method (CEM)
    This implementation is basically Algorithm 3.3. in [1] with the addition of decreasing noise [2].
    CEM is closely related to PoWER. The most significant difference is that the importance sampels are not kept over
    iterations and that the covariance matrix is not scaled with the returns, thus allowing for negative returns.

    .. seealso::
        [1] P.T. de Boer, D.P. Kroese, S. Mannor, R.Y. Rubinstein, "A Tutorial on the Cross-Entropy Method",
        Annals OR, 2005

        [2] I. Szita, A. Lörnicz, "Learning Tetris Using the NoisyCross-Entropy Method", Neural Computation, 2006
    """

    name: str = 'cem'

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
                 extra_expl_std_init: float = 2.,
                 extra_expl_decay_iter: int = 10,
                 full_cov: bool = False,
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
        :param num_is_samples: number of samples (policy parameter sets & returns) for importance sampling,
                               indirectly specifies the performance quantile $1 - \rho$ [1]
        :param expl_std_init: initial standard deviation for the exploration strategy
        :param expl_std_min: minimal standard deviation for the exploration strategy
        :param full_cov: pass `True` to compute a full covariance matrix for sampling the next policy parameter values,
                         else a diagonal covariance is used
        :param extra_expl_std_init: additional standard deviation for the parameter exploration added to the diagonal
                                    entries of the covariance matirx.
        :param extra_expl_decay_iter: limit for the linear decay of the additional standard deviation, i.e. last
                                      iteration in which the additional exploration noise is applied
        :param symm_sampling: use an exploration strategy which samples symmetric populations
        :param base_seed: seed added to all other seeds in order to make the experiments distinct but repeatable
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not extra_expl_std_init >= 0:
            raise pyrado.ValueErr(given=extra_expl_std_init, ge_constraint='0')
        if not extra_expl_decay_iter > 0:
            raise pyrado.ValueErr(given=extra_expl_decay_iter, g_constraint='0')

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
            full_cov=full_cov,
            std_init=expl_std_init,
            std_min=expl_std_min,
        )
        if symm_sampling:
            # Exploration strategy based on symmetrical normally distributed noise
            if self.pop_size%2 != 0:
                # Symmetric buffer needs to have an even number of samples
                self.pop_size += 1
            self._expl_strat = SymmParamExplStrat(self._expl_strat)

        self.num_is_samples = min(pop_size, num_is_samples)
        self.best_policy_param_values = policy.param_values.clone()
        if isinstance(self._expl_strat.noise, DiagNormalNoise):
            self.extra_expl_std_init = to.ones_like(self._policy.param_values)*extra_expl_std_init
        elif isinstance(self._expl_strat.noise, FullNormalNoise):
            self.extra_expl_std_init = to.eye(self._policy.num_param)*extra_expl_std_init
        else:
            raise NotImplementedError  # CEM could also sample using different distributions

    @to.no_grad()
    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float = None):
        # Average the return values over the rollouts
        rets_avg_ros = to.tensor(param_results.mean_returns)

        # Descending sort according to return values and the importance samples a.k.a. elites (see [1, p.12])
        idcs_dcs = to.argsort(rets_avg_ros, descending=True)
        idcs_dcs = idcs_dcs[:self.num_is_samples]
        rets_avg_is = rets_avg_ros[idcs_dcs]
        params_is = param_results.parameters[idcs_dcs, :]

        # Store the very best policy parameter values for saving it later
        self.best_policy_param_values = params_is[0, :].clone()

        # Update the policy parameters from the mean importance samples
        self._policy.param_values = to.mean(params_is, dim=0)

        # Update the exploration covariance from the empirical variance of the importance samples
        if isinstance(self._expl_strat.noise, DiagNormalNoise):
            std_is = to.std(params_is, dim=0)
            extra_expl_std = self.extra_expl_std_init*to.max(1. - self._curr_iter/self.extra_expl_std_init,
                                                             to.zeros_like(self.extra_expl_std_init))  # see [2, p.4]
            self._expl_strat.noise.adapt(std=std_is + extra_expl_std)
        elif isinstance(self._expl_strat.noise, FullNormalNoise):
            cov_is = cov(params_is, data_along_rows=True)
            extra_expl_cov = to.pow(self.extra_expl_std_init, 2)*to.max(1. - self._curr_iter/self.extra_expl_std_init,
                                                                        to.zeros_like(self.extra_expl_std_init))  # see [2, p.4]
            self._expl_strat.noise.adapt(cov=cov_is + extra_expl_cov)
        else:
            raise NotImplementedError  # CEM could also sample using different distributions

        # Logging
        self.logger.add_value('median imp samp return', to.median(rets_avg_is))
        self.logger.add_value('min imp samp return', to.min(rets_avg_is))
        self.logger.add_value('min expl strat std', to.min(self._expl_strat.std))
        self.logger.add_value('avg expl strat std', to.mean(self._expl_strat.std.data).detach().numpy())
        self.logger.add_value('max expl strat std', to.max(self._expl_strat.std))
        self.logger.add_value('expl strat entropy', self._expl_strat.get_entropy().item())

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Reset memory for importance sampling
        self.best_policy_param_values = self._policy.param_values.clone()

    def save_snapshot(self, meta_info: dict = None):
        # Algorithm.save_snapshot() saves the policy used for exploring, we override it here with the best policy
        super().save_snapshot(meta_info)

        max_ret_policy = deepcopy(self._policy)
        max_ret_policy.param_values = self.best_policy_param_values

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            to.save(max_ret_policy, osp.join(self._save_dir, 'policy.pt'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            if 'prefix' in meta_info and 'suffix' in meta_info:
                to.save(max_ret_policy, osp.join(self._save_dir,
                                                 f"{meta_info['prefix']}_policy_{meta_info['suffix']}.pt"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                to.save(max_ret_policy, osp.join(self._save_dir, f"{meta_info['prefix']}_policy.pt"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                to.save(max_ret_policy, osp.join(self._save_dir, f"policy_{meta_info['suffix']}.pt"))
            else:
                raise NotImplementedError
