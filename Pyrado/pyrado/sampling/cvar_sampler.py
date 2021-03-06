import numpy as np
from typing import List

from pyrado.logger.step import LoggerAware
from pyrado.sampling.sampler import SamplerBase
from pyrado.sampling.step_sequence import StepSequence


def select_cvar(rollouts, epsilon: float, gamma: float = 1.):
    """
    Select a subset of rollouts so that their mean discounted return is the CVaR(eps) of the full rollout set.

    :param rollouts: list of rollouts
    :param epsilon: chosen return quantile
    :param gamma: discount factor to compute the discounted return, default is 1 (no discount)
    :return: list of selected rollouts
    """
    # Select epsilon-quantile of returns
    # Do this first since it is easier here, even though we have to compute the returns again
    # To do so, we first sort the paths by their returns
    rollouts.sort(key=lambda r: r.discounted_return(gamma))

    # Now, computing the quantile on a sorted list is easy
    k = len(rollouts) * epsilon
    n = int(np.floor(k))

    # Only use the selected paths
    return rollouts[0:n]


class CVaRSampler(SamplerBase, LoggerAware):
    """
    Samples rollouts to optimize the CVaR of the discounted return.
    This is done by sampling more rollouts, and then only using the epsilon-qunatile of them.
    """

    def __init__(self,
                 wrapped_sampler,
                 epsilon: float,
                 gamma: float = 1.,
                 *,
                 min_rollouts: int = None,
                 min_steps: int = None):
        """
        Constructor

        :param wrapped_sampler: the inner sampler used to sample the full data set
        :param epsilon: quantile of rollouts that will be kept
        :param gamma: discount factor to compute the discounted return, default is 1 (no discount)
        :param min_rollouts: minimum number of complete rollouts to sample
        :param min_steps: minimum total number of steps to sample
        """
        self._wrapped_sampler = wrapped_sampler
        self.epsilon = epsilon
        self.gamma = gamma

        # Call SamplerBase constructor
        super().__init__(min_rollouts=min_rollouts, min_steps=min_steps)

    def set_min_count(self, min_rollouts=None, min_steps=None):
        # Set inner sampler's parameter values (back) to the user-specified number of rollouts / steps
        super().set_min_count(min_rollouts=min_rollouts, min_steps=min_steps)

        # Increase the number of rollouts / steps that will be sampled since we will discard (1 - eps) quantile
        # This modifies the inner samplers parameter values, that's ok since we don't use them afterwards
        if min_rollouts is not None:
            # Expand rollout count to full set
            min_rollouts = int(min_rollouts / self.epsilon)
        if min_steps is not None:
            # Simply increasing the number of steps as done for the rollouts is not identical, however it is goo enough
            min_steps = int(min_steps / self.epsilon)
        self._wrapped_sampler.set_min_count(min_rollouts=min_rollouts, min_steps=min_steps)

    def reinit(self, env=None, policy=None):
        # Delegate to inner sampler
        self._wrapped_sampler.reinit(env=env, policy=policy)

    def sample(self) -> List[StepSequence]:
        # Sample full data set
        fullset = self._wrapped_sampler.sample()

        # Log return-based metrics using the full data set
        rets = [ro.undiscounted_return() for ro in fullset]
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_std = np.std(rets)
        self.logger.add_value('full num rollouts', len(fullset))
        self.logger.add_value('full avg rollout len', np.mean([ro.length for ro in fullset]))
        self.logger.add_value('full avg return', ret_avg)
        self.logger.add_value('full median return', ret_med)
        self.logger.add_value('full std return', ret_std)

        # Return subset
        return select_cvar(fullset, self.epsilon, self.gamma)
