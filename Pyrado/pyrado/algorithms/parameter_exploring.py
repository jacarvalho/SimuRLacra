import torch as to
import numpy as np
import joblib
import os.path as osp
from abc import abstractmethod

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parameter_exploration_sampler import ParameterExplorationSampler, ParameterSamplingResult
from pyrado.exploration.stochastic_params import StochasticParamExplStrat
from pyrado.utils.input_output import print_cbt


class ParameterExploring(Algorithm):
    """ Base for all algorithms that explore directly in the policy parameter space """

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 max_iter: int,
                 num_rollouts: int,
                 pop_size: int = None,
                 num_sampler_envs: int = 4,
                 base_seed: int = None,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_rollouts: number of rollouts per solution
        :param pop_size: number of solutions in the population
        :param num_sampler_envs: number of parallel environments in the sampler
        :param base_seed: seed added to all other seeds in order to make the experiments distinct but repeatable
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        assert isinstance(pop_size, int) and pop_size > 0 or pop_size is None

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        # Store the inputs
        self._env = env
        self.num_rollouts = num_rollouts

        # Auto-select population size if needed
        if pop_size is None:
            pop_size = 4 + int(3*np.log(policy.num_param))
            print_cbt(f'Initialized population size to {pop_size}.', 'y')
        self.pop_size = pop_size

        # Create sampler
        self.sampler = ParameterExplorationSampler(
            env,
            policy,
            num_envs=num_sampler_envs,
            num_rollouts_per_param=num_rollouts,
            seed=base_seed
        )

        # Stopping criterion
        self.ret_avg_stack = 1e3*np.random.randn(20)  # stack size = 20
        self.thold_ret_std = 1e-1  # algorithm terminates if below for multiple iterations

        # Set this in subclasses!
        self._expl_strat = None

    @property
    def expl_strat(self) -> StochasticParamExplStrat:
        return self._expl_strat

    def stopping_criterion_met(self) -> bool:
        """
        Check if the average reward of the mean policy did not change more than the specified threshold over the
        last iterations.
        """
        return False
        # if np.std(self.ret_avg_stack) < self.thold_ret_std:
        #     return True
        # else:
        #     return False

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Sample new policy parameters
        paramsets = self._expl_strat.sample_param_sets(
            self._policy.param_values,
            self.pop_size,
            include_nominal_params=True
        )

        with to.no_grad():
            # Sample rollouts using these parameters
            rollouts = self.sampler.sample(paramsets)

        # Evaluate the mean policy
        ret_avg_curr = rollouts[0].mean_undiscounted_return
        param_results = rollouts[1:]

        # Store the average return for the stopping criterion
        self.ret_avg_stack = np.delete(self.ret_avg_stack, 0)
        self.ret_avg_stack = np.append(self.ret_avg_stack, ret_avg_curr)

        all_rets = param_results.mean_returns
        all_lengths = np.array([len(ro)
                                for pg in param_results
                                for ro in pg.rollouts])

        # Log metrics computed from the old policy (before the update)
        self.logger.add_value('curr policy return', ret_avg_curr)
        self.logger.add_value('max return', float(np.max(all_rets)))
        self.logger.add_value('median return', float(np.median(all_rets)))
        self.logger.add_value('avg return', float(np.mean(all_rets)))
        self.logger.add_value('avg rollout len', float(np.mean(all_lengths)))
        self.logger.add_value('min mag policy param',
                              self._policy.param_values[to.argmin(abs(self._policy.param_values))])
        self.logger.add_value('max mag policy param',
                              self._policy.param_values[to.argmax(abs(self._policy.param_values))])

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(np.mean(all_rets)), meta_info)

        # Update the policy
        self.update(param_results, ret_avg_curr)

    @abstractmethod
    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float):
        """
        Update the policy from the given samples.

        :param param_results: Sampled parameters with evaluation
        :param ret_avg_curr: Average return for the current parameters
        """
        raise NotImplementedError

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            joblib.dump(self._env, osp.join(self._save_dir, 'env.pkl'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            pass

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir
        super().load_snapshot(ld, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._env = joblib.load(osp.join(ld, 'env.pkl'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            pass
