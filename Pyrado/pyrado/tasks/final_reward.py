import numpy as np
from colorama import Style
from tabulate import tabulate
from typing import NamedTuple

import pyrado
from pyrado.tasks.base import Task, TaskWrapper
from pyrado.utils import get_class_name


class FinalRewMode(NamedTuple):
    """ The specification of how the final state should be rewarded or punished """
    state_dependent: bool = False
    time_dependent: bool = False
    always_positive: bool = False
    always_negative: bool = False

    def __str__(self):
        """ Get an information string. """
        return Style.BRIGHT + f'{get_class_name(self)}' + Style.RESET_ALL + f' (id: {id(self)})\n' + \
               tabulate(
                   [['state_dependent', self.state_dependent], ['time_dependent', self.time_dependent],
                    ['always_positive', self.always_positive], ['always_negative', self.always_negative]]
               )


class FinalRewTask(TaskWrapper):
    """
    Wrapper for tasks which yields a reward / cost on success / failure
    
    :usage:
    .. code-block:: python

        task = FinalRewTask(DesStateTask(spec, state_des, rew_fcn, success_fcn), mode=FinalRewMode(), factor=1e3)
    """

    def __init__(self, wrapped_task: Task, mode: FinalRewMode, factor: float = 1e3):
        """
        Constructor

        :param wrapped_task: task to wrap
        :param mode: mode for calculating the final reward
        :param factor: value to scale the final reward, does not matter if `mode.time_dependent is True`
        """
        # Call TaskWrapper's constructor
        super().__init__(wrapped_task)

        if not isinstance(mode, FinalRewMode):
            raise pyrado.TypeErr(given=mode, expected_type=FinalRewMode)
        self.mode = mode
        self.factor = factor
        self._yielded_final_rew = False

    @property
    def yielded_final_rew(self) -> bool:
        """ Get the flag that signals if this instance already yielded its final reward. """
        return self._yielded_final_rew

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._yielded_final_rew = False

    def compute_final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        """
        Compute the reward / cost on task completion / fail of this task.

        :param state: current state of the environment
        :param remaining_steps: number of time steps left in the episode
        :return: final reward of this task
        """
        if self._yielded_final_rew:
            # Only yield the final reward once
            return 0.

        else:
            self._yielded_final_rew = True

            # Default case
            if (not self.mode.always_positive and not self.mode.always_negative and
                not self.mode.state_dependent and not self.mode.time_dependent):
                if self.has_failed(state):
                    return -1.*np.abs(self.factor)
                else:
                    return np.abs(self.factor)

            elif (self.mode.always_positive and not self.mode.always_negative and
                  not self.mode.state_dependent and not self.mode.time_dependent):
                if self.has_failed(state):
                    return 0.
                else:
                    return np.abs(self.factor)

            elif (not self.mode.always_positive and self.mode.always_negative and
                  not self.mode.state_dependent and not self.mode.time_dependent):
                if self.has_failed(state):
                    return -1.*np.abs(self.factor)
                else:
                    return 0.

            elif (self.mode.always_positive and not self.mode.always_negative and
                  self.mode.state_dependent and not self.mode.time_dependent):
                if self.has_failed(state):
                    return 0.
                else:
                    act = np.zeros(self.env_spec.act_space.shape)  # dummy
                    step_rew = self._wrapped_task.step_rew(state, act, remaining_steps)
                    return self.factor*abs(step_rew)

            elif (not self.mode.always_positive and self.mode.always_negative and
                  self.mode.state_dependent and not self.mode.time_dependent):
                if self.has_failed(state):
                    act = np.zeros(self.env_spec.act_space.shape)  # dummy
                    step_rew = self._wrapped_task.step_rew(state, act, remaining_steps)
                    return -1.*self.factor*abs(step_rew)
                else:
                    return 0.

            elif (not self.mode.always_positive and not self.mode.always_negative and
                  self.mode.state_dependent and not self.mode.time_dependent):
                act = np.zeros(self.env_spec.act_space.shape)  # dummy
                step_rew = self._wrapped_task.step_rew(state, act, remaining_steps)
                if self.has_failed(state):
                    return -1.*self.factor*abs(step_rew)
                else:
                    return self.factor*abs(step_rew)

            elif (not self.mode.always_positive and not self.mode.always_negative and
                  self.mode.state_dependent and self.mode.time_dependent):
                act = np.zeros(self.env_spec.act_space.shape)  # dummy
                step_rew = self._wrapped_task.step_rew(state, act, remaining_steps)
                if self.has_failed(state):
                    return -1.*remaining_steps*abs(step_rew)
                else:
                    return remaining_steps*abs(step_rew)

            elif (not self.mode.always_positive and not self.mode.always_negative and
                  not self.mode.state_dependent and self.mode.time_dependent):
                if self.has_failed(state):
                    return -1.*remaining_steps
                else:
                    return remaining_steps

            elif (self.mode.always_positive and not self.mode.always_negative and
                  not self.mode.state_dependent and self.mode.time_dependent):
                if self.has_failed(state):
                    return 0.
                else:
                    return remaining_steps

            elif (not self.mode.always_positive and self.mode.always_negative and
                  not self.mode.state_dependent and self.mode.time_dependent):
                if self.has_failed(state):
                    return -1.*remaining_steps
                else:
                    return 0.

            else:
                raise NotImplementedError(f'No matching configuration found for the given FinalRewMode:\n{self.mode}')


class BestStateFinalRewTask(TaskWrapper):
    """
    Wrapper for tasks which yields a reward / cost on success / failure based on the best reward / cost observed in the
    current trajectory that is scaled by the number of taken / remaining time steps.
    """

    def __init__(self, wrapped_task: Task, max_steps: int, factor: float = 1.):
        """
        Constructor

        :param wrapped_task: task to wrap
        :param max_steps: maximum number of time steps in the environment to infer the number of steps when done
        :param factor: value to scale the final reward
        """
        # Call TaskWrapper's constructor
        super().__init__(wrapped_task)

        if not isinstance(max_steps, int):
            raise pyrado.TypeErr(given=max_steps, expected_type=int)
        self._max_steps = max_steps
        self.factor = factor
        self.best_rew = -pyrado.inf
        self._yielded_final_rew = False

    @property
    def yielded_final_rew(self) -> bool:
        """ Get the flag that signals if this instance already yielded its final reward. """
        return self._yielded_final_rew

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        rew = self._wrapped_task.step_rew(state, act, remaining_steps)
        if rew > self.best_rew:
            self.best_rew = rew
        return rew

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.best_rew = -pyrado.inf
        self._yielded_final_rew = False

    def compute_final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        """
        Compute the reward / cost on task completion / fail of this task.

        :param state: current state of the environment
        :param remaining_steps: number of time steps left in the episode
        :return: final reward of this task
        """
        if self._yielded_final_rew:
            # Only yield the final reward once
            return 0.

        else:
            self._yielded_final_rew = True

            # Return the highest reward / lowest cost scaled with the number of taken time steps and the factor
            scale = (self._max_steps - remaining_steps) * self.factor
            return scale * self.best_rew
