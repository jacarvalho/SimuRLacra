import joblib
import numpy as np
import os.path as osp
import torch as to
from abc import ABC, abstractmethod
from typing import Sequence

import pyrado
from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.base import Algorithm
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.policies.base import Policy
from pyrado.sampling.step_sequence import StepSequence


class ActorCritic(Algorithm, ABC):
    """ Base class of all actor critic algorithms """

    def __init__(self,
                 env: Env,
                 actor: Policy,
                 critic: [to.nn.Module, Policy, GAE],
                 save_dir: str,
                 max_iter: int,
                 logger: StepLogger = None):
        """
        Constructor

        :param env: environment which the policy operates
        :param actor: policy taking the actions in the environment
        :param critic: estimates the value of states (e.g. advantage or return)
        :param save_dir: directory to save the snapshots i.e. the results in
        :param max_iter: maximum number of iterations
        :param logger: logger for every step of the algorithm
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, actor, logger)

        # Store the inputs
        self._env = env
        self._critic = critic

        # Initialize
        self._expl_strat = None
        self.sampler = None
        self._lr_scheduler = None
        self._lr_scheduler_hparam = None

    @property
    def critic(self) -> [to.nn.Module, Policy, GAE]:
        """ Get the critic. """
        return self._critic

    @critic.setter
    def critic(self, critic: [to.nn.Module, Policy, GAE]):
        """ Set the critic. """
        if not isinstance(critic, (to.nn.Module, GAE)):
            pyrado.TypeErr(given=critic, expected_type=[to.nn.Module, Policy, GAE])
        self._critic = critic

    @property
    def expl_strat(self) -> NormalActNoiseExplStrat:
        return self._expl_strat

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Sample rollouts
        ros = self.sampler.sample()

        # Log return-based metrics
        rets = [ro.undiscounted_return() for ro in ros]
        ret_min = np.min(rets)
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_max = np.max(rets)
        ret_std = np.std(rets)
        self.logger.add_value('max return', ret_max)
        self.logger.add_value('median return', ret_med)
        self.logger.add_value('avg return', ret_avg)
        self.logger.add_value('min return', ret_min)
        self.logger.add_value('std return', ret_std)
        self.logger.add_value('num rollouts', len(ros))
        self.logger.add_value('avg rollout len', np.mean([ro.length for ro in ros]))

        # Update the advantage estimator and the policy
        self.update(ros)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

    @abstractmethod
    def update(self, rollouts: Sequence[StepSequence]):
        """
        Update the actor and critic parameters from the given batch of rollouts.

        :param rollouts: batch of rollouts
        """
        raise NotImplementedError

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize sampler in case env or policy changed
        self.sampler.reinit()

        # Reset the critic (also resets its learning rate scheduler)
        self.critic.reset()

        # Reset the learning rate scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.last_epoch = -1

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            joblib.dump(self._env, osp.join(self._save_dir, 'env.pkl'))
            to.save(self._expl_strat.policy, osp.join(self._save_dir, 'policy.pt'))
            to.save(self._critic, osp.join(self._save_dir, 'critic.pt'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            if 'prefix' in meta_info and 'suffix' in meta_info:
                to.save(self._expl_strat.policy, osp.join(self._save_dir,
                                                          f"{meta_info['prefix']}_policy_{meta_info['suffix']}.pt"))
                to.save(self._critic, osp.join(self._save_dir,
                                               f"{meta_info['prefix']}_critic_{meta_info['suffix']}.pt"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                to.save(self._expl_strat.policy, osp.join(self._save_dir, f"{meta_info['prefix']}_policy.pt"))
                to.save(self._critic, osp.join(self._save_dir, f"{meta_info['prefix']}_critic.pt"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                to.save(self._expl_strat.policy, osp.join(self._save_dir, f"policy_{meta_info['suffix']}.pt"))
                to.save(self._critic, osp.join(self._save_dir, f"critic_{meta_info['suffix']}.pt"))
            else:
                raise NotImplementedError

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir
        super().load_snapshot(ld, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._env = joblib.load(osp.join(ld, 'env.pkl'))
            self._critic.load_state_dict(to.load(osp.join(ld, 'critic.pt')).state_dict())
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            if 'prefix' in meta_info and 'suffix' in meta_info:
                self._critic.load_state_dict(
                    to.load(osp.join(ld, f"{meta_info['prefix']}_critic_{meta_info['suffix']}.pt")).state_dict()
                )
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                self._critic.load_state_dict(
                    to.load(osp.join(ld, f"{meta_info['prefix']}_critic.pt")).state_dict()
                )
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                self._critic.load_state_dict(
                    to.load(osp.join(ld, f"critic_{meta_info['suffix']}.pt")).state_dict()
                )
            else:
                raise NotImplementedError
