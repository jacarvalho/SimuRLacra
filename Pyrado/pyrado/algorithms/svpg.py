import joblib
import numpy as np
import os.path as osp
import torch as to
from copy import deepcopy
from scipy.spatial.distance import squareform, pdist
from torch.distributions.kl import kl_divergence
from typing import Sequence

import pyrado
from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import compute_action_statistics
from pyrado.environments.base import Env
from pyrado.policies.fnn import FNNPolicy
from pyrado.algorithms.advantage import ValueFunctionSpace
from pyrado.utils.data_types import EnvSpec
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.sampling.step_sequence import StepSequence


class SVPGParticle(Policy):
    """
    An actor-critic particle of type `Policy`.
    Particles' parameters are considered to be part of the parameter distribution optimized by the SVPG algorithm.
    """

    def __init__(self, spec: EnvSpec, actor: Policy, critic: GAE):
        """
        Constructor

        :param spec: specification of the environment the particle should act in
        :param actor_hparam: hyper-parameters for the actor, e.g. `hidden_sizes` and `hidden_nonlin`
        :param value_fcn_hparam: hyper-parameters for the critic's V-function, e.g. `hidden_sizes` and `hidden_nonlin`
        :param critic_hparam: hyper-parameters for critic, e.g. `gamma` and `lamda`
        """

        # Call Policy's constructor
        super().__init__(spec)

        # Create actor and critic
        self.actor = actor
        self.critic = critic

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        """
        Initializes the parameters of the actor and critic networks with given values.

        :param init_values: the initial values for the actor and critic networks
        """
        self.actor.init_param(init_values, **kwargs)
        self.critic.value_fcn.init_param(init_values, **kwargs)

    def value(self, obs: to.Tensor) -> to.Tensor:
        """
        Predict the value of a given observation. Forwards to the critic head.

        :param obs: the observation
        :return: the predicted value
        """
        return self.critic.values(obs)

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Get the action given an observation. Forwards to the actor head.

        :param obs: the observation
        :return: the action
        """
        return self.actor(obs)


class SVPG(Algorithm):
    """
    Stein Variational Policy Gradient (SVPG)

    .. seealso::
        [1] Yang Liu, Prajit Ramachandran, Qiang Liu, Jian Peng, "Stein Variational Policy Gradient", arXiv, 2017
    """

    name: str = 'svpg'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 particle_hparam: dict,
                 max_iter: int,
                 num_particles: int,
                 temperature: float,
                 lr: float,
                 horizon: int,
                 std_init: float = 1.0,
                 min_rollouts: int = None,
                 min_steps: int = 10000,
                 num_sampler_envs: int = 4,
                 serial: bool = True,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param particle_hparam: hyper-parameters for particle template construction
        :param max_iter: number of iterations
        :param num_particles: number of distinct particles
        :param temperature: the temperature of the SVGD determines how jointly the training takes place
        :param lr: the learning rate for the update of the particles
        :param horizon: horizon for each particle
        :param std_init: initial standard deviation for the exploration
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param num_sampler_envs: number of environments for parallel sampling
        :param serial: serial mode can be switched off which can be used to partly control the flow of SVPG from outside
        :param logger: logger for every step of the algorithm
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(particle_hparam, dict):
            raise pyrado.TypeErr(given=particle_hparam, expected_type=dict)
        if not all([key in particle_hparam for key in ['actor', 'value_fcn', 'critic']]):
            raise AttributeError

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy=None, logger=logger)

        # Store the inputs
        self._env = env
        self.num_particles = num_particles
        self.horizon = horizon
        self.lr = lr
        self.temperature = temperature
        self.serial = serial

        # Prepare placeholders for particles
        self.particles = [None] * num_particles
        self.particleSteps = [None] * num_particles
        self.expl_strats = [None] * num_particles
        self.optimizers = [None] * num_particles
        self.fixed_particles = [None] * num_particles
        self.fixed_expl_strats = [None] * num_particles
        self.samplers = [None] * num_particles
        self.count = 0
        self.update_count = 0

        # Particle factory
        actor = FNNPolicy(spec=env.spec, **particle_hparam['actor'])
        value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **particle_hparam['value_fcn'])
        critic = GAE(value_fcn, **particle_hparam['critic'])
        particle = SVPGParticle(env.spec, actor, critic)

        for i in range(self.num_particles):
            self.particles[i] = deepcopy(particle)
            self.particles[i].init_param()
            self.expl_strats[i] = NormalActNoiseExplStrat(self.particles[i].actor, std_init)
            self.optimizers[i] = to.optim.Adam(self.expl_strats[i].parameters(), lr=self.lr)
            self.fixed_particles[i] = deepcopy(self.particles[i])
            self.fixed_expl_strats[i] = deepcopy(self.expl_strats[i])
            self.particleSteps[i] = 0

            if self.serial:
                self.samplers[i] = ParallelSampler(
                    env, self.expl_strats[i], num_sampler_envs, min_rollouts=min_rollouts, min_steps=min_steps
                )

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Serial flag must not be set when interacting through step and reset
        if not self.serial:
            raise NotImplementedError('Step cannot be called if serial flag is False!')

        self.count += 1
        ros_all_particles, rets_all_particles = [], []
        for i in range(self.num_particles):
            ros_one_particle = self.samplers[i].sample()
            # ro_concat = StepSequence.concat(ros_one_particle)
            # ro_concat.torch(data_type=to.get_default_dtype())
            ros_all_particles.append(ros_one_particle)
            rets_all_particles.append(np.array([ro.undiscounted_return() for ro in ros_one_particle]))
            self.particleSteps[i] =+ 1
            if self.horizon != 0 and (self.particleSteps[i] > self.horizon):
                self.particles[i].init_param()
                self.particleSteps[i] = 0


        # Logging
        num_ros_all_prtcls = np.array([len(p) for p in ros_all_particles])
        len_ros_all_prtcls = np.array([np.mean([ro.length for ro in p]) for p in ros_all_particles])
        avg_rets_all_prtcls = np.array([np.mean(p) for p in rets_all_particles])
        median_rets_all_prtcls = np.array([np.median(p) for p in rets_all_particles])
        max_rets_all_prtcls = np.array([np.max(p) for p in rets_all_particles])
        std_rets_all_prtcls = np.array([np.std(p) for p in rets_all_particles])
        avg_explstrat_stds = np.array([to.mean(e.noise.std.data).item() for e in self.expl_strats])
        self.logger.add_value('num rollouts pp', num_ros_all_prtcls)
        self.logger.add_value('avg rollout len pp', len_ros_all_prtcls)
        self.logger.add_value('avg return pp', avg_rets_all_prtcls)
        self.logger.add_value('median return pp', median_rets_all_prtcls)
        self.logger.add_value('max return pp', max_rets_all_prtcls)
        self.logger.add_value('std return pp', std_rets_all_prtcls)
        self.logger.add_value('avg expl strat stds pp', avg_explstrat_stds)

        # Logging for recording (averaged over particles
        self.logger.add_value('num rollouts', np.mean(num_ros_all_prtcls))
        self.logger.add_value('avg rollout len', np.mean(len_ros_all_prtcls))
        self.logger.add_value('avg return', np.mean(avg_rets_all_prtcls))
        self.logger.add_value('median return', np.median(median_rets_all_prtcls))
        self.logger.add_value('std return', np.mean(std_rets_all_prtcls))
        self.logger.record_step()

        self.update(ros_all_particles)

    def kernel(self, X: to.Tensor) -> (to.Tensor, to.Tensor):
        """
        Compute the RBF-kernel and the corresponding derivatives.

        :param X: the tensor to compute the kernel from
        :return: the kernel and its derivatives
        """
        X_np = X.cpu().data.numpy()  # use numpy because torch median is flawed
        pairwise_dists = squareform(pdist(X_np))**2

        # Median trick
        h = np.median(pairwise_dists)
        h = np.sqrt(0.5*h/np.log(self.num_particles + 1))

        # Compute RBF Kernel
        Kxx = to.exp(-to.from_numpy(pairwise_dists).to(to.get_default_dtype())/h**2/2)

        # Compute kernel gradient
        dx_Kxx = -(Kxx).matmul(X)
        sum_Kxx = Kxx.sum(1)
        for i in range(X.shape[1]):
            dx_Kxx[:, i] = dx_Kxx[:, i] + X[:, i].matmul(sum_Kxx)
        dx_Kxx /= (h**2)

        return Kxx, dx_Kxx

    def update(self, rollouts: Sequence[StepSequence]):
        r"""
        Train the particles $mu$.

        :param rollouts: rewards collected from the rollout
        """
        policy_grads = []
        parameters = []

        for i in range(self.num_particles):
            # Get the rollouts associated to the i-th particle
            concat_ros = StepSequence.concat(rollouts[i])
            concat_ros.torch()

            act_stats = compute_action_statistics(concat_ros, self.expl_strats[i])
            act_stats_fixed = compute_action_statistics(concat_ros, self.fixed_expl_strats[i])

            klds = to.distributions.kl_divergence(act_stats.act_distr, act_stats_fixed.act_distr)
            entropy = act_stats.act_distr.entropy()
            log_prob = act_stats.log_probs

            concat_ros.rewards = concat_ros.rewards - (0.1*klds.mean(1)).view(-1) - 0.1*entropy.mean(1).view(-1)

            # Update the advantage estimator's parameters and return advantage estimates
            adv = self.particles[i].critic.update(rollouts[i], use_empirical_returns=True)

            # Estimate policy gradients
            self.optimizers[i].zero_grad()
            policy_grad = -to.mean(log_prob*adv.detach())
            policy_grad.backward()  # step comes later than usual

            # Collect flattened parameter and gradient vectors
            policy_grads.append(self.expl_strats[i].param_grad)
            parameters.append(self.expl_strats[i].param_values)

        parameters = to.stack(parameters)
        policy_grads = to.stack(policy_grads)
        Kxx, dx_Kxx = self.kernel(parameters)
        grad_theta = (to.mm(Kxx, policy_grads/self.temperature) + dx_Kxx)/self.num_particles

        for i in range(self.num_particles):
            self.expl_strats[i].param_grad = grad_theta[i]
            self.optimizers[i].step()
        self.update_count += 1

    def save_snapshot(self, meta_info: dict = None):
        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            joblib.dump(self._env, osp.join(self._save_dir, 'env.pkl'))
            for idx, p in enumerate(self.particles):
                to.save(p, osp.join(self._save_dir, f'particle_{idx}.pt'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            for idx, p in enumerate(self.particles):
                to.save(p, osp.join(self._save_dir, f'particle_{idx}.pt'))

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            for idx in range(self.num_particles):
                self.particles[idx].load_state_dict(
                    to.load(osp.join(ld, f'particle_{idx}.pt')).state_dict()
                )
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            raise NotImplementedError
