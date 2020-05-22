import joblib
import os.path as osp

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environment_wrappers.utils import typed_env
from pyrado.sampling.cvar_sampler import CVaRSampler


class EPOpt(Algorithm):
    """
    Ensemble Policy Optimization (EPOpt)

    This algorithm wraps another algorithm on a very shallow level. It replaces the subroutine's sampler with a
    CVaRSampler, but does not have its own logger.

    .. seealso::
        [1] A. Rajeswaran, S. Ghotra, B. Ravindran, S. Levine, "EPOpt: Learning Robust Neural Network Policies using
        Model Ensembles", ICLR, 2017
    """

    name: str = 'epopt'

    def __init__(self,
                 subroutine: Algorithm,
                 skip_iter: int,
                 epsilon: float,
                 gamma: float = 1.):
        """
        Constructor

        :param subroutine: algorithm which performs the policy / value-function optimization
        :param skip_iter: number of iterations for which all rollouts will be used (see prefix full)
        :param epsilon: quantile of rollouts that will be kept
        :param gamma: discount factor to compute the discounted return, default is 1 (no discount)
        """
        if not isinstance(subroutine, Algorithm):
            raise pyrado.TypeErr(given=subroutine, expected_type=Algorithm)
        if not typed_env(subroutine.sampler.env, DomainRandWrapperLive):  # there is a domain randomization wrapper
            raise pyrado.TypeErr(given=subroutine.sampler.env, expected_type=DomainRandWrapperLive)

        # Call Algorithm's constructor with the subroutine's properties
        super().__init__(subroutine.save_dir, subroutine.max_iter, subroutine.policy, subroutine.logger)

        # Store inputs
        self._subroutine = subroutine
        self.epsilon = epsilon
        self.gamma = gamma
        self.skip_iter = skip_iter

        # Override the subroutine's sampler
        self._subroutine.sampler = CVaRSampler(
                self._subroutine.sampler,
                epsilon=1.,  # keep all rollouts until curr_iter = skip_iter
                gamma=self.gamma,
                min_rollouts=self._subroutine.sampler.min_rollouts,
                min_steps=self._subroutine.sampler.min_steps,
        )

        # Save initial randomizer
        joblib.dump(subroutine.sampler.randomizer, osp.join(self.save_dir, 'randomizer.pkl'))

    @property
    def subroutine(self) -> Algorithm:
        return self._subroutine

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Activate the CVaR mechanism after skip_iter iterations
        if self.curr_iter == self.skip_iter:
            self._subroutine.sampler.epsilon = self.epsilon

        # Call subroutine
        self._subroutine.step(snapshot_mode, meta_info)

    def save_snapshot(self, meta_info: dict = None):
        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            if self.curr_iter == self.skip_iter:
                # Save the last snapshot before applying the CVaR
                self._subroutine.save_snapshot(meta_info=dict(prefix=f'iter_{self.skip_iter}'))
            else:
                self._subroutine.save_snapshot(meta_info=None)
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._subroutine.load_snapshot(ld, meta_info)
        else:
            raise pyrado.ValueErr(msg=f'{self.name} is not supposed be run as a subroutine!')
