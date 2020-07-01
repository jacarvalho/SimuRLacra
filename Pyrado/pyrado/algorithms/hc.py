import numpy as np
import torch as to

import pyrado
from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.environments.base import Env
from pyrado.policies.base import Policy
from pyrado.utils.input_output import print_cbt
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult
from pyrado.exploration.stochastic_params import HyperSphereParamNoise, NormalParamNoise
from abc import abstractmethod


class HC(ParameterExploring):
    """
    Hill Climbing (HC)

    HC is a heuristic-based policy search method that samples a population of policy parameters per iteration
    and evaluates them on multiple rollouts. If one of the new parameters is better than the current one it is kept.
    If the exploration parameters grow too large, they are reset.
    """

    name: str = 'hc'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 max_iter: int,
                 num_rollouts: int,
                 expl_factor: float,
                 pop_size: int = None,
                 base_seed: int = None,
                 num_sampler_envs: int = 4):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_rollouts: number of rollouts per policy sample
        :param expl_factor: scalar value which determines how the exploration strategy adapts its search space
        :param pop_size: number of solutions in the population
        :param base_seed: seed added to all other seeds in order to make the experiments distinct but repeatable
        :param num_sampler_envs: number of environments for parallel sampling
        """
        # Call ParameterExploring's constructor
        super().__init__(
            save_dir,
            env,
            policy,
            max_iter,
            num_rollouts,
            pop_size=pop_size,
            base_seed=base_seed,
            num_sampler_envs=num_sampler_envs
        )

        # Store the inputs
        self.expl_factor = float(expl_factor)

        # Parameters for reset heuristics
        self.max_policy_param = 1e3

    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float):
        # Average the return values over the rollouts
        rets_avg_ros = param_results.mean_returns

        # Update the policy
        if np.max(rets_avg_ros) > ret_avg_curr:
            # Update the policy parameters to the best solution from the current population
            idx_max = np.argmax(rets_avg_ros)
            self._policy.param_values = param_results[idx_max].params

        # Re-initialize the policy parameters if the became too large
        if (to.abs(self._policy.param_values) > self.max_policy_param).any():
            self._policy.init_param()
            print_cbt('Reset policy parameters.', 'y')

        # Update exploration strategy in subclass
        self.update_expl_strat(rets_avg_ros, ret_avg_curr)

    @abstractmethod
    def update_expl_strat(self, rets_avg_ros: np.ndarray, ret_avg_curr: float):
        raise NotImplementedError


class HCNormal(HC):
    """ Hill Climbing variant using an exploration strategy with normally distributed noise on the policy parameters """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        :param expl_std_init: initial standard deviation for the exploration strategy
        :param args: forwarded the superclass constructor
        :param kwargs: forwarded the superclass constructor
        """
        # Preprocess inputs and call HC's constructor
        expl_std_init = kwargs.pop('expl_std_init')
        if 'expl_r_init' in kwargs:
            # This is just for the ability to create one common hyper-param list for HCNormal and HCHyper
            kwargs.pop('expl_r_init')

        # Get from kwargs with default values
        expl_std_min = kwargs.pop('expl_std_min', 0.01)

        # Call HC's constructor
        super().__init__(*args, **kwargs)

        self._expl_strat = NormalParamNoise(
            param_dim=self._policy.num_param,
            std_init=expl_std_init,
            std_min=expl_std_min,
        )

    def update_expl_strat(self, rets_avg_ros: np.ndarray, ret_avg_curr: float):
        # Update the exploration distribution
        if np.max(rets_avg_ros) > ret_avg_curr:
            self._expl_strat.adapt(std=self._expl_strat.std/self.expl_factor)
        elif np.max(rets_avg_ros) < ret_avg_curr:
            self._expl_strat.adapt(std=self._expl_strat.std*self.expl_factor)
        else:
            pass  # don't change if the current policy parameter set has been the best sample

        self.logger.add_value('min expl strat std', to.min(self._expl_strat.std))
        self.logger.add_value('avg expl strat std', to.mean(self._expl_strat.std.data).detach().numpy())
        self.logger.add_value('max expl strat std', to.max(self._expl_strat.std))
        self.logger.add_value('expl strat entropy', np.mean(self._expl_strat.get_entropy().detach().numpy()))


class HCHyper(HC):
    """ Hill Climbing variant using an exploration strategy that samples policy parameters from a hyper-sphere """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        :param expl_r_init: initial radius of the hyper sphere for the exploration strategy
        :param args: forwarded the superclass constructor
        :param kwargs: forwarded the superclass constructor
        """
        # Preprocess inputs and call HC's constructor
        expl_r_init = kwargs.pop('expl_r_init')
        if expl_r_init <= 0:
            raise pyrado.ValueErr(given=expl_r_init, g_constraint='0')

        if 'expl_std_init' in kwargs:
            # This is just for the ability to create one common hyper-param list for HCNormal and HCHyper
            kwargs.pop('expl_std_init')

        # Get from kwargs with default values
        self.expl_r_min = kwargs.pop('expl_r_min', 0.01)
        self.expl_r_max = max(expl_r_init, kwargs.pop('expl_r_max', 10.))

        # Call HC's constructor
        super().__init__(*args, **kwargs)

        self._expl_strat = HyperSphereParamNoise(
            param_dim=self._policy.num_param,
            expl_r_init=expl_r_init,
        )

    def update_expl_strat(self, rets_avg_ros: np.ndarray, ret_avg_curr: float):
        # Update the exploration strategy
        if np.max(rets_avg_ros) > ret_avg_curr:
            self._expl_strat.adapt(r=self._expl_strat.r/self.expl_factor)
        elif np.max(rets_avg_ros) < ret_avg_curr:
            self._expl_strat.adapt(r=self._expl_strat.r*self.expl_factor)
        else:
            pass  # don't change if the current policy parameter set has been the best sample

        # Re-initialize the exploration parameters if the became too small or too large
        if self._expl_strat.r < self.expl_r_min or self._expl_strat.r > self.expl_r_max:
            self._expl_strat.reset_expl_params()

        self.logger.add_value('smallest expl param', self._expl_strat.r)
        self.logger.add_value('largest expl param', self._expl_strat.r)
