import functools
import torch as to
from copy import deepcopy
from torch.distributions import Distribution
from typing import NamedTuple

import pyrado
from pyrado.sampling.step_sequence import StepSequence
from pyrado.exploration.stochastic_action import StochasticActionExplStrat
from pyrado.utils.input_output import print_cbt


class ActionStatistics(NamedTuple):
    """
    act_distr: probability distribution at the given policy output values
    log_probs: $\log (p(act|obs, hidden))$ if hidden exists, else $\log (p(act|obs))$
    entropy: entropy of the action distribution
    """
    act_distr: Distribution
    log_probs: to.Tensor
    entropy: to.Tensor


def compute_action_statistics(steps: StepSequence, expl_strat: StochasticActionExplStrat) -> ActionStatistics:
    r"""
    Get the action distribution from the exploration strategy, compute the log action probabilities and entropy
    for the given rollout using the given exploration strategy.

    .. note::
        Requires the exploration strategy to have a (most likely custom) `evaluate()` method.

    :param steps: recorded rollout data
    :param expl_strat: exploration strategy used to generate the data
    :return: collected action statistics, see `ActionStatistics`
    """
    # Evaluate rollout(s)
    distr = expl_strat.evaluate(steps)

    # Collect results
    return ActionStatistics(distr, distr.log_prob(steps.actions), distr.entropy())


def until_thold_exceeded(thold: float, max_iter: int = None):
    """
    Designed to wrap a function and repeat it until the return value exceeds a threshold.

    :param thold: threshold
    :param max_iter: maximum number of iterations of the wrapped function, set to `None` to run the loop relentlessly
    :return: wrapped function
    """
    def actual_decorator(trn_eval_fcn):
        """
        Designed to wrap a training + evaluation function and repeat it  it until the return value exceeds a threshold.

        :param trn_eval_fcn: function to wrap
        :return: wrapped function
        """
        @functools.wraps(trn_eval_fcn)
        def wrapper_trn_eval_fcn(*args, **kwargs):
            ret = -pyrado.inf
            cnt_iter = 0
            while ret <= thold:  # <= guarantees that we at least train once, even if thold is -inf
                # Train and evaluate
                ret = trn_eval_fcn(*args, **kwargs)
                cnt_iter += 1
                # Check if done
                if ret < thold:
                    print_cbt(f'The policy did not exceed the threshold {thold}.', 'y', True)
                if max_iter is None:
                    print_cbt(f'Repeating training and evaluation ...', 'y', True)
                else:
                    if cnt_iter < max_iter:
                        print_cbt(f'Repeating training and evaluation ...', 'y', True)
                    else:
                        print_cbt(f'Exiting the training and evaluation loop after {max_iter} iterations.', 'y', True)
            return ret

        return wrapper_trn_eval_fcn

    return actual_decorator


class ReplayMemory:
    """ Base class for storing step transitions """

    def __init__(self, capacity: int):
        """
        Constructor

        :param capacity: number of steps a.k.a. transitions in the memory
        """
        self.capacity = int(capacity)
        self._memory = None

    @property
    def memory(self) -> StepSequence:
        """ Get the replay buffer. """
        return self._memory

    @property
    def isempty(self) -> bool:
        """ Check if the replay buffer is empty. """
        return self._memory is None

    def __len__(self) -> int:
        """ Get the number of transitions stored in the buffer. """
        return self._memory.length

    def push(self, ros: [list, StepSequence], truncate_last: bool = True):
        """
        Save a sequence of steps and drop of steps if the capacity is exceeded.

        :param ros: list of rollouts or one concatenated rollout
        :param truncate_last: remove the last step from each rollout, forwarded to `StepSequence.concat`
        """
        if isinstance(ros, list):
            # Concatenate given rollouts if necessary
            ros = StepSequence.concat(ros)
        elif isinstance(ros, StepSequence):
            pass
        else:
            pyrado.TypeErr(given=ros, expected_type=[list, StepSequence])

        # Add new steps
        if self.isempty:
            self._memory = deepcopy(ros)  # on the very first call
        else:
            self._memory = StepSequence.concat([self._memory, ros], truncate_last=truncate_last)

        num_surplus = self._memory.length - self.capacity
        if num_surplus > 0:
            # Drop surplus of old steps
            self._memory = self._memory[num_surplus:]

    def sample(self, batch_size: int) -> tuple:
        """
        Sample randomly from the replay memory.

        :param batch_size: number of samples
        :return: tuple of transition steps and associated next steps
        """
        return self._memory.sample_w_next(batch_size)

    def reset(self):
        self._memory = None

    def avg_reward(self) -> float:
        """
        Compute the average reward for all steps stored in the replay memory.

        :return: average reward
        """
        if self._memory is None:
            raise pyrado.TypeErr(msg='The replay memory is empty!')
        else:
            return sum(self._memory.rewards)/self._memory.length
