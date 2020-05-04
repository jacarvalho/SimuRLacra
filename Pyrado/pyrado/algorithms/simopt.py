import joblib
import numpy as np
import os
import os.path as osp
import torch as to
from abc import ABC

import pyrado
from pyrado.algorithms.actor_critic import ActorCritic
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import until_thold_exceeded
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.environments.quanser.base import RealEnv
from pyrado.policies.base import Policy
from pyrado.sampling.bootstrapping import bootstrap_ci
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.sampling.rollout import rollout
from pyrado.utils.input_output import print_cbt
from pyrado.utils.math import UnitCubeProjector
from pyrado.utils.standardizing import standardize


class SimOpt(Algorithm, ABC):
    # TODO: this class is work in progress...
    """
    SimOpt

    .. see also::
        Y. Chebotar et. al., "Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience"
    """

    name: str = 'simopt'

    def __init__(self,
                 save_dir: str,
                 env_sim: MetaDomainRandWrapper,
                 env_real: [RealEnv, EnvWrapper],
                 subroutine: Algorithm,
                 max_iter: int,
                 thold_succ: float,
                 init_cand: to.Tensor = None,
                 warmstart: bool = False,
                 policy_param_init: to.Tensor = None,
                 critic_param_init: to.Tensor = None
                 ):
        """
        Constructor
        """
        assert isinstance(env_sim, MetaDomainRandWrapper)
        assert isinstance(subroutine, Algorithm)

        # Call Algorithm's constructor without specifying the policy
        super().__init__(save_dir, max_iter, subroutine.policy)

        self._env_sim = env_sim
        self._env_real = env_real
        self._subroutine = subroutine
        self.cand = init_cand
        self.thold_succ = to.tensor([thold_succ], dtype=to.get_default_dtype())
        self.warmstart = warmstart
        self.policy_param_init = policy_param_init.detach() if policy_param_init is not None else None
        self.critic_param_init = critic_param_init.detach() if critic_param_init is not None else None

        self.W = np.ones(env_real.obs_space.flat_dim())  # importance weights of each observation dimension
        self.w1 = 1.  # l1 norm weight
        self.w2 = 1.  # l2 norm weight

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """ TODO """
        prefix = f'iter_{self._curr_iter}'

        # Train a policy using the subroutine (saves to iter_{self._curr_iter}_policy.pt)
        wrapped_trn_fcn = until_thold_exceeded(self.thold_succ.item(), max_iter=3)(self.train_policy_sim)
        wrapped_trn_fcn(self.cand, prefix)

        # Evaluate the current policy on the target domain
        policy = to.load(osp.join(self._save_dir, f'{prefix}_policy.pt'))
        input('[Press Enter to connect to the target system ...] ')
        print_cbt(f'Evaluating {prefix}_policy on the target system ...', 'c', bright=True)

        obs_real = rollout(self._env_real, policy, eval=True).observations

        def _obj_fnc(cand):
            self._env_sim.adapt_randomizer(cand.numpy())
            obs_sim = rollout(self._env_sim, policy, eval=True).observations
            weighted_diff = self.W * (obs_sim - obs_real)
            c1 = np.sum(np.linalg.norm(weighted_diff, ord=1, axis=1))
            c2 = np.sum(np.linalg.norm(weighted_diff, ord=2, axis=1))
            return self.w1 * c1 + self.w2 * c2

        # Optimize the simulation parameter distribution (obj_fnc)
        # [...]

        self.cand = ...
        self._env_sim.adapt_randomizer(self.cand.numpy())


    def train_policy_sim(self, cand: to.Tensor, prefix: str) -> float:
        """
        Train a policy in simulation for given hyper-parameters from the domain randomizer.

        :param cand: hyper-parameters for the domain parameter distribution coming from the domain randomizer
        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        """
        # Save the individual candidate
        to.save(cand, osp.join(self._save_dir, f'{prefix}_candidate.pt'))

        # Set the domain randomizer given the hyper-parameters
        self._env_sim.adapt_randomizer(cand.numpy())

        # Reset the subroutine's algorithm which includes resetting the exploration
        self._subroutine.reset()

        if not self.warmstart or self._curr_iter == 0:
            # Reset the subroutine's policy (and value function)
            self._subroutine.policy.init_param(self.policy_param_init)
            if isinstance(self._subroutine, ActorCritic):
                self._subroutine.critic.value_fcn.init_param(self.critic_param_init)
            if self.policy_param_init is None:
                print_cbt('Learning the new solution from scratch', 'y')
            else:
                print_cbt('Learning the new solution given an initialization', 'y')

        elif self.warmstart and self._curr_iter > 0:
            # Continue from the previous policy (and value function)
            self._subroutine.policy.load_state_dict(
                to.load(osp.join(self._save_dir, f'iter_{self._curr_iter - 1}_policy.pt')).state_dict()
            )
            if isinstance(self._subroutine, ActorCritic):
                self._subroutine.critic.value_fcn.load_state_dict(
                    to.load(osp.join(self._save_dir, f'iter_{self._curr_iter - 1}_valuefcn.pt')).state_dict()
                )
            print_cbt(f'Initialized the new solution with the results from iteration {self._curr_iter - 1}', 'y')

        # Train a policy in simulation using the subroutine
        self._subroutine.train(snapshot_mode='best', meta_info=dict(prefix=prefix))

        # Return the average return of the trained policy in simulation
        sampler = ParallelSampler(self._env_sim, self._subroutine.policy, num_envs=8, min_rollouts=80)
        ros = sampler.sample()
        avg_ret_sim = np.mean([ro.undiscounted_return() for ro in ros])
        return float(avg_ret_sim)
