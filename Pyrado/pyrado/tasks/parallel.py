import numpy as np
from copy import deepcopy
from typing import Sequence

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.tasks.base import Task
from pyrado.utils.input_output import print_cbt


class ParallelTasks(Task):
    """ Task class for a set of tasks a.k.a. goals which can be achieved in any order or parallel """

    def __init__(self,
                 tasks: Sequence[Task],
                 hold_rew_when_done: bool = False,
                 allow_failures: bool = False,
                 verbose: bool = False):
        """
        Constructor

        :param tasks: sequence of tasks a.k.a. goals, the order matters
        :param hold_rew_when_done: if `True` reward values for done tasks will be stored and added every step
        :param allow_failures: if `True` this allows to continue after one sub-task failed, by default `False`
        :param verbose: print messages on task completion

        .. note::
            This task can also be wrapped by a `FinalRewTask` to enjoy modularity.

            `hold_rew_when_done=True` only makes sense for positive rewards.
        """
        self._tasks = deepcopy(tasks)
        self.succeeded_tasks = np.full(len(self), False, dtype=bool)
        self.failed_tasks = np.full(len(self), False, dtype=bool)
        self.hold_rew_when_done = hold_rew_when_done
        if self.hold_rew_when_done:
            self.held_rews = np.zeros(len(self))
        self.allow_failures = allow_failures
        self.verbose = verbose

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def env_spec(self) -> EnvSpec:
        return self._tasks[0].env_spec  # safe to assume that all tasks have the same env_spec

    @property
    def tasks(self) -> Sequence[Task]:
        """ Get the list of tasks. """
        return deepcopy(self._tasks)

    @property
    def state_des(self) -> list:
        """ Get a list of all desired states. """
        return [task.state_des for task in self._tasks]

    @state_des.setter
    def state_des(self, states_des: Sequence):
        """ Set all desired states from a list of desired states. """
        if not len(states_des) == len(self.space_des):
            raise pyrado.ShapeErr(given=states_des, expected_match=self.space_des)
        for task in self._tasks:
            task.state_des = states_des[i]

    @property
    def space_des(self) -> list:
        """ Get a list of all desired spaces. """
        return [task.space_des for task in self._tasks]

    @space_des.setter
    def space_des(self, spaces_des: Sequence):
        """ Set all desired spaces from a list of desired spaces. """
        if not len(spaces_des) == len(self.space_des):
            raise pyrado.ShapeErr(given=spaces_des, expected_match=self.space_des)
        for task in self._tasks:
            task.space_des = spaces_des[i]

    @property
    def rew_fcn(self) -> list:
        """ Get a list of all reward functions. """
        return [task.rew_fcn for task in self._tasks]

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        """ Get the step reward accumulated from every non-done task. """
        step_rew = 0.
        for i in range(len(self)):
            if not (self.succeeded_tasks[i] or self.failed_tasks[i]):
                # Task has not been marked done yet
                step_rew += self._tasks[i].step_rew(state, act, remaining_steps)
            else:
                # Task is done
                if self.hold_rew_when_done:
                    # Add the last reward from every done task (also true for failed tasks)
                    step_rew += self.held_rews[i]

        # Check if any task is done and update the
        final_rew = self._is_any_task_done(state, act, remaining_steps)  # zero if the task is not done

        return step_rew + final_rew

    def compute_final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        """
        Compute the reward / cost on task completion / fail of this task.
        Since this task holds multiple sub-tasks, the final reward / cost is computed for them, too.

        .. note::
            The `ParallelTasks` class is not a subclass of `TaskWrapper`, i.e. this function only looks at the
            immediate sub-tasks.

        :param state: current state of the environment
        :param remaining_steps: number of time steps left in the episode
        :return: final reward of all sub-tasks
        """
        sum_final_rew = 0.
        for t in self._tasks:
            sum_final_rew += t.compute_final_rew(state, remaining_steps)
        return sum_final_rew

    def reset(self, **kwargs):
        """ Reset all tasks. """
        for task in self._tasks:
            task.reset(**kwargs)

        # Reset internal check list for done tasks
        self.succeeded_tasks = np.full(len(self), False, dtype=bool)
        self.failed_tasks = np.full(len(self), False, dtype=bool)

        # Reset the stored reward values for done tasks
        if self.hold_rew_when_done:
            self.held_rews = np.zeros(len(self))

    def _is_any_task_done(self,
                          state: np.ndarray,
                          act: np.ndarray,
                          remaining_steps: int,
                          verbose: bool = False) -> float:
        """
        Check if any of the tasks is done. If so, return the final reward of this task.

        :param state: current state
        :param act: current action
        :param remaining_steps: number of time steps left in the episode
        """
        task_final_rew = 0.
        for i, task in enumerate(self._tasks):
            if not self.succeeded_tasks[i] and not self.failed_tasks[i] and task.is_done(state):
                # Task has not been marked done yet, but is now done

                if task.has_succeeded(state):
                    # Check off successfully completed tasks
                    self.succeeded_tasks[i] = True
                    if verbose:
                        print_cbt(f'task {i} has succeeded (is done) at state {state}', 'g')

                elif task.has_failed(state):
                    # Check off unsuccessfully completed tasks
                    self.failed_tasks[i] = True
                    if verbose:
                        print_cbt(f'Task {i} has failed (is done) at state {state}', 'r')

                else:
                    raise pyrado.ValueErr(msg=f'Task {i} neither succeeded or failed but is done!')

                # Give a reward for completing the task defined by the task
                task_final_rew += task.final_rew(state, remaining_steps)  # there could be more than one finished task

                if self.hold_rew_when_done:
                    # Memorize current reward (only becomes active after success or fail)
                    self.held_rews[i] = self._tasks[i].step_rew(state, act, remaining_steps)

        return task_final_rew

    def has_succeeded(self, state: np.ndarray = None) -> bool:
        """
        Check if this tasks is done. The ParallelTasks is successful if all sub-tasks are successful.

        :param state: environments current state
        :return: `True` if succeeded
        """
        successful = np.all(self.succeeded_tasks)
        if successful and self.verbose:
            print_cbt(f'All {len(self)} parallel sub-tasks are done successfully', 'g')
        return successful

    def has_failed(self, state: np.ndarray = None) -> bool:
        """ Check if this task has failed. """
        if self.allow_failures:
            # The ParallelTasks fail if all of them fail
            return np.all(self.failed_tasks)
        else:
            # The ParallelTasks fail if one of them fails
            return np.any(self.failed_tasks)
