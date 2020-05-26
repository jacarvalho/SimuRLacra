import joblib
import numpy as np
import os.path as osp
import sys
import torch as to
from copy import deepcopy
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import ReplayMemory
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import EpsGreedyExplStrat
from pyrado.logger.step import StepLogger, ConsolePrinter, CSVPrinter
from pyrado.policies.fnn import DiscrActQValFNNPolicy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.utils.input_output import print_cbt


class DQL(Algorithm):
    """
    Deep Q-Learning (without bells and whistles)

    .. seealso::
        [1] V. Mnih et.al., "Human-level control through deep reinforcement learning", Nature, 2015
    """

    name: str = 'dql'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: DiscrActQValFNNPolicy,
                 memory_size: int,
                 eps_init: float,
                 eps_schedule_gamma: float,
                 gamma: float,
                 max_iter: int,
                 num_batch_updates: int,
                 target_update_intvl: int = 5,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 batch_size: int = 256,
                 num_sampler_envs: int = 4,
                 max_grad_norm: float = 0.5,
                 lr: float = 5e-4,
                 lr_scheduler=None,
                 lr_scheduler_hparam: [dict, None] = None,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: environment which the policy operates
        :param policy: (current) Q-network updated by this algorithm
        :param memory_size: number of transitions in the replay memory buffer
        :param eps_init: initial value for the probability of taking a random action, constant if `eps_schedule_gamma==1`
        :param eps_schedule_gamma: temporal discount factor for the exponential decay of epsilon
        :param gamma: temporal discount factor for the state values
        :param max_iter: number of iterations (policy updates)
        :param num_batch_updates: number of batch updates per algorithm steps
        :param target_update_intvl: number of iterations that pass before updating the target network
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param batch_size: number of samples per policy update batch
        :param num_sampler_envs: number of environments for parallel sampling
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param logger: logger for every step of the algorithm
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(policy, DiscrActQValFNNPolicy):
            raise pyrado.TypeErr(given=policy, expected_type=DiscrActQValFNNPolicy)

        if logger is None:
            # Create logger that only logs every 100 steps of the algorithm
            logger = StepLogger(print_interval=100)
            logger.printers.append(ConsolePrinter())
            logger.printers.append(CSVPrinter(osp.join(save_dir, 'progress.csv')))

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        # Store the inputs
        self._env = env
        self.target = deepcopy(self._policy)
        self.target.eval()  # will not be trained using the optimizer
        self._memory_size = memory_size
        self.eps = eps_init
        self.gamma = gamma
        self.target_update_intvl = target_update_intvl
        self.num_batch_updates = num_batch_updates
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Initialize
        self._expl_strat = EpsGreedyExplStrat(self._policy, eps_init, eps_schedule_gamma)
        self._memory = ReplayMemory(memory_size)
        self.sampler = ParallelSampler(
            env, self._expl_strat,
            num_envs=1,
            min_steps=min_steps,
            min_rollouts=min_rollouts
        )
        self.sampler_eval = ParallelSampler(
            env, self._policy,
            num_envs=num_sampler_envs,
            min_steps=100*env.max_steps,
            min_rollouts=None
        )
        self.optim = to.optim.RMSprop([{'params': self._policy.parameters()}], lr=lr)
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

    @property
    def expl_strat(self) -> EpsGreedyExplStrat:
        return self._expl_strat

    @property
    def memory(self) -> ReplayMemory:
        """ Get the replay memory. """
        return self._memory

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Sample steps and store them in the replay memory
        ros = self.sampler.sample()
        self._memory.push(ros)

        while len(self._memory) < self.memory.capacity:
            # Warm-up phase
            print_cbt('Collecting samples until replay memory contains if full.', 'w')
            # Sample steps and store them in the replay memory
            ros = self.sampler.sample()
            self._memory.push(ros)

        # Log return-based metrics
        if self._curr_iter % self.logger.print_interval == 0:
            ros = self.sampler_eval.sample()
            rets = [ro.undiscounted_return() for ro in ros]
            ret_max = np.max(rets)
            ret_med = np.median(rets)
            ret_avg = np.mean(rets)
            ret_min = np.min(rets)
            ret_std = np.std(rets)
        else:
            ret_max, ret_med, ret_avg, ret_min, ret_std = 5*[-pyrado.inf]  # dummy values
        self.logger.add_value('max return', np.round(ret_max, 4))
        self.logger.add_value('median return', np.round(ret_med, 4))
        self.logger.add_value('avg return', np.round(ret_avg, 4))
        self.logger.add_value('min return', np.round(ret_min, 4))
        self.logger.add_value('std return', np.round(ret_std, 4))
        self.logger.add_value('avg rollout length', np.round(np.mean([ro.length for ro in ros]), 2))
        self.logger.add_value('num rollouts', len(ros))
        self.logger.add_value('avg memory reward', np.round(self._memory.avg_reward(), 4))

        # Use data in the memory to update the policy and the target Q-function
        self.update()

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

    def loss_fcn(self, q_vals: to.Tensor, expected_q_vals: to.Tensor) -> to.Tensor:
        r"""
        The Huber loss function on the one-step TD error $\delta = Q(s,a) - (r + \gamma \max_a Q(s^\prime, a))$.

        :param q_vals: state-action values $Q(s,a)$, from policy network
        :param expected_q_vals: expected state-action values $r + \gamma \max_a Q(s^\prime, a)$, from target network
        :return: loss value
        """
        return to.nn.functional.smooth_l1_loss(q_vals, expected_q_vals)

    def update(self):
        """ Update the policy's and target Q-function's parameters on transitions sampled from the replay memory. """
        losses = to.zeros(self.num_batch_updates)
        policy_grad_norm = to.zeros(self.num_batch_updates)

        for b in tqdm(range(self.num_batch_updates), total=self.num_batch_updates,
                      desc=f'Updating', unit='batches', file=sys.stdout, leave=False):

            # Sample steps and the associated next step from the replay memory
            steps, next_steps = self._memory.sample(self.batch_size)
            steps.torch(data_type=to.get_default_dtype())
            next_steps.torch(data_type=to.get_default_dtype())

            # Create masks for the non-final observations
            not_done = to.tensor(1. - steps.done, dtype=to.get_default_dtype())

            # Compute the state-action values Q(s,a) using the current DQN policy
            q_vals = self.expl_strat.policy.q_values_chosen(steps.observations)

            # Compute the second term of TD-error
            next_v_vals = self.target.q_values_chosen(next_steps.observations).detach()
            expected_q_val = steps.rewards + not_done*self.gamma*next_v_vals

            # Compute the loss, clip the gradients if desired, and do one optimization step
            loss = self.loss_fcn(q_vals, expected_q_val)
            losses[b] = loss.data
            self.optim.zero_grad()
            loss.backward()
            policy_grad_norm[b] = self.clip_grad(self.expl_strat.policy, self.max_grad_norm)
            self.optim.step()

            # Update the target network by copying all weights and biases from the DQN policy
            if (self._curr_iter*self.num_batch_updates + b)%self.target_update_intvl == 0:
                self.target.load_state_dict(self.expl_strat.policy.state_dict())

        # Schedule the exploration parameter epsilon
        self.expl_strat.schedule_eps(self._curr_iter)

        # Update the learning rate if a scheduler has been specified
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        # Logging
        with to.no_grad():
            self.logger.add_value('loss after', to.mean(losses).item())
        self.logger.add_value('expl strat eps', self.expl_strat.eps.item())
        self.logger.add_value('avg policy grad norm', to.mean(policy_grad_norm).item())
        if self._lr_scheduler is not None:
            self.logger.add_value('learning rate', self._lr_scheduler.get_lr())

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This instance is not a subroutine of a meta-algorithm
            joblib.dump(self._env, osp.join(self._save_dir, 'env.pkl'))
            to.save(self.target, osp.join(self._save_dir, 'target.pt'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            if 'prefix' in meta_info and 'suffix' in meta_info:
                to.save(self.target,
                        osp.join(self._save_dir, f"{meta_info['prefix']}_target_{meta_info['suffix']}.pt"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                to.save(self.target, osp.join(self._save_dir, f"{meta_info['prefix']}_target.pt"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                to.save(self.target, osp.join(self._save_dir, f"target_{meta_info['suffix']}.pt"))
            else:
                raise NotImplementedError

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir
        super().load_snapshot(ld, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._env = joblib.load(osp.join(ld, 'env.pkl'))
            self.target.load_state_dict(to.load(osp.join(ld, 'target.pt')).state_dict())
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            if 'prefix' in meta_info and 'suffix' in meta_info:
                self.target.load_state_dict(
                    to.load(osp.join(ld, f"{meta_info['prefix']}_target_{meta_info['suffix']}.pt")).state_dict()
                )
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                self.target.load_state_dict(
                    to.load(osp.join(ld, f"{meta_info['prefix']}_target.pt")).state_dict()
                )
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                self.target.load_state_dict(
                    to.load(osp.join(ld, f"target_{meta_info['suffix']}.pt")).state_dict()
                )
            else:
                raise NotImplementedError

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize sampler in case env or policy changed
        self.sampler.reinit()

        # Reset the replay memory
        self._memory.reset()

        # Reset the learning rate scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.last_epoch = -1
