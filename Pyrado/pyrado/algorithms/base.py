import os.path as osp
import pandas as pd
import torch as to
from abc import ABC, abstractmethod
from typing import Any

import pyrado
from pyrado.logger import get_log_prefix_dir
from pyrado.logger.step import StepLogger, LoggerAware
from pyrado.policies.base import Policy
from pyrado import set_seed
from pyrado.utils import get_class_name
from pyrado.utils.input_output import print_cbt


class Algorithm(ABC, LoggerAware):
    """
    Base class of all algorithms in Pyrado
    Algorithms specify the way how the policy is updated as well as the exploration strategy used to acquire samples.
    """

    name: str = None  # unique identifier
    iteration_key: str = 'iteration'

    def __init__(self, save_dir: str, max_iter: int, policy: [Policy, None], logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param max_iter: maximum number of iterations
        :param policy: Pyrado policy (subclass of PyTorch's Module) to train
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(max_iter, int) and max_iter > 0:
            raise pyrado.ValueErr(given=max_iter, g_constraint='0')
        assert isinstance(policy, Policy) or policy is None

        if save_dir is None:
            save_dir = get_log_prefix_dir()
        self._save_dir = save_dir
        self._max_iter = max_iter
        self._curr_iter = 0
        self._policy = policy
        self._logger = logger
        self._highest_avg_ret = -pyrado.inf  # for snapshot_mode = 'best'

    @property
    def save_dir(self) -> str:
        """ Get the directory where the data is saved to. """
        return self._save_dir

    @property
    def max_iter(self) -> int:
        """ Get the maximum number of iterations. """
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int):
        """ Set the maximum number of iterations. """
        assert max_iter > 0
        self._max_iter = max_iter

    @property
    def curr_iter(self) -> int:
        """ Get the current iteration counter. """
        return self._curr_iter

    @property
    def policy(self):
        """ Get the algorithm's policy. """
        return self._policy

    @property
    def expl_strat(self):
        """ Get the algorithm's exploration strategy. """
        return None

    def stopping_criterion_met(self) -> bool:
        """
        Checks if one of the algorithms (characteristic) stopping criterions is met.

        .. note::
            This function can be overwritten by the subclasses to implement custom stopping behavior.

        :return: flag if one of the stopping criterion(s) is met
        """
        return False

    def reset(self, seed: int = None):
        """
        Reset the algorithm to it's initial state. This should NOT reset learned policy parameters.
        By default, this resets the iteration count. 
        Be sure to call this function if you override it.

        :param seed: seed value for the random number generators, pass None for no seeding
        """
        # Reset the exploration strategy if any
        if self.expl_strat is not None:
            self.expl_strat.reset_expl_params()

        # Reset internal variables
        self._curr_iter = 0
        self._highest_avg_ret = -pyrado.inf

        if seed is not None:
            set_seed(seed)
            print_cbt(f"Set the random number generators' seed to {seed}.", 'y')

    def train(self,
              load_dir: str = None,
              snapshot_mode: str = 'latest',
              seed: int = None,
              meta_info: dict = None):
        """
        Train one/multiple policy/policies in a given environment.

        :param load_dir: if not `None` the training snapshot will be loaded from the given directory, i.e. the training
                         does not start from scratch
        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new high-score)
        :param seed: seed value for the random number generators, pass None for no seeding
        :param meta_info: is not None if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        if load_dir is not None:
            self.load_snapshot(load_dir)
            print_cbt(f'Loaded the snapshot from {load_dir}.', 'g', bright=True)

        if self._policy is not None:
            print_cbt(f'{get_class_name(self)} started training a {get_class_name(self._policy)} '
                      f'with {self._policy.num_param} parameters using the snapshot mode {snapshot_mode}.', 'g')
            # Set dropout and batch normalization layers to training mode
            self._policy.train()
        else:
            print_cbt(f'{get_class_name(self)} started training using the snapshot mode {snapshot_mode}.', 'g')

        if seed is not None:
            set_seed(seed)
            print_cbt(f"Set the random number generators' seed to {seed}.", 'y')

        while self._curr_iter < self.max_iter and not self.stopping_criterion_met():
            # Record current iteration to logger
            self.logger.add_value(self.iteration_key, self._curr_iter)

            # Acquire data, save the training progress, and update the parameters
            self.step(snapshot_mode, meta_info)

            # Update logger and print
            self.logger.record_step()

            # Increase the iteration counter
            self._curr_iter += 1

            print(self._policy.param_values)

        if self.stopping_criterion_met():
            stopping_reason = 'Stopping criterion met!'
        else:
            stopping_reason = 'Maximum number of iterations reached!'

        if self._policy is not None:
            print_cbt(f'{get_class_name(self)} finished training a {get_class_name(self._policy)} '
                      f'with {self._policy.num_param} parameters. {stopping_reason}', 'g')
            # Set dropout and batch normalization layers to evaluation mode
            self._policy.eval()
        else:
            print_cbt(f'{get_class_name(self)} finished training. {stopping_reason}', 'g')

    @abstractmethod
    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Perform a single iteration of the algorithm. This includes collecting the data, updating the parameters, and
        adding the metrics of interest to the logger. Does not update the `curr_iter` attribute.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any):
        """ Update the policy's (and value functions') parameters based on the collected rollout data. """
        raise NotImplementedError

    @staticmethod
    def clip_grad(module: to.nn.Module, max_grad_norm: [float, None]) -> float:
        """
        Clip all gradients of the provided Module (e.g., a policy or an advantage estimator) by their L2 norm value.

        .. note::
            The gradient clipping has to be applied between loss.backward() and optimizer.step()

        :param module: Module containing parameters
        :param max_grad_norm: maximum L2 norm for the gradient
        :return: total norm of the parameters (viewed as a single vector)
        """
        if max_grad_norm is not None:
            to.nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm, norm_type=2)  # returns unclipped norm

        # Calculate the clipped gradient's L2 norm (for logging)
        total_norm = 0.
        for p in list(filter(lambda p: p.grad is not None, module.parameters())):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
        return total_norm**0.5

    def make_snapshot(self, snapshot_mode: str, curr_avg_ret: float = None, meta_info: dict = None):
        """
        Make a snapshot of the training progress.
        This method is called from the subclasses and delegates to the custom method `save_snapshot()`.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param curr_avg_ret: current average return used for the snapshot_mode 'best' to trigger `save_snapshot()`
        :param meta_info: is not None if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        if snapshot_mode == 'latest':
            self.save_snapshot(meta_info)
        elif snapshot_mode == 'best':
            if curr_avg_ret is None:
                raise pyrado.ValueErr(msg="curr_avg_ret must not be None when snapshot_mode = 'best'!")
            if curr_avg_ret > self._highest_avg_ret:
                self._highest_avg_ret = curr_avg_ret
                self.save_snapshot(meta_info)
        elif snapshot_mode in {'no', 'none'}:
            pass  # don't save anything
        else:
            raise pyrado.ValueErr(given=snapshot_mode, eq_constraint="'latest', 'best', or 'no'")

    def save_snapshot(self, meta_info: dict = None):
        """
        Save the algorithm information (e.g., environment, policy, ect.).
        Subclasses should call the base method to save the policy.

        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a `dict` of information about the current iteration of the meta-algorithm
        """
        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            to.save(self._policy, osp.join(self._save_dir, 'policy.pt'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            if 'prefix' in meta_info and 'suffix' in meta_info:
                to.save(self._policy, osp.join(self._save_dir,
                                               f"{meta_info['prefix']}_policy_{meta_info['suffix']}.pt"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                to.save(self._policy, osp.join(self._save_dir, f"{meta_info['prefix']}_policy.pt"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                to.save(self._policy, osp.join(self._save_dir, f"policy_{meta_info['suffix']}.pt"))
            else:
                raise NotImplementedError

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        """
        Load the algorithm information (e.g., environment, policy, ect.).
        Subclasses should call the base method to load the policy.

        :param load_dir: explicit directory to load from, if `None` (default) `self._save_dir` is used
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a `dict` of information about the current iteration of the meta-algorithm
        """
        ld = load_dir if load_dir is not None else self._save_dir
        try:
            self._curr_iter = pd.read_csv(osp.join(ld, 'progress.csv'))[self.iteration_key].iloc[-1]
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self._curr_iter = 0

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._policy.load_state_dict(to.load(osp.join(ld, 'policy.pt')).state_dict())
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            if 'prefix' in meta_info and 'suffix' in meta_info:
                self._policy.load_state_dict(
                    to.load(osp.join(ld, f"{meta_info['prefix']}_policy_{meta_info['suffix']}.pt")).state_dict()
                )
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                self._policy.load_state_dict(
                    to.load(osp.join(ld, f"{meta_info['prefix']}_policy.pt")).state_dict()
                )
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                self._policy.load_state_dict(
                    to.load(osp.join(ld, f"policy_{meta_info['suffix']}.pt")).state_dict()
                )
            else:
                raise NotImplementedError
