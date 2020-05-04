from pyrado.algorithms.base import Algorithm
from pyrado.environment_wrappers.adversarial import AdversarialDynamicsWrapper, AdversarialStateWrapper, \
    AdversarialObservationWrapper
from pyrado.environment_wrappers.state_augmentation import StateAugmentationWrapper
from pyrado.environments.sim_base import SimEnv
from pyrado.exploration.stochastic_action import StochasticActionExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.sampling.sequences import *


class ARPL(Algorithm):
    """
    Adversarially Robust Policy Learning (ARPL)

    .. seealso::
        A. Mandlekar, Y. Zhu, A. Garg, L. Fei-Fei, S. Savarese, "Adversarially Robust Policy Learning:
        Active Construction of Physically-Plausible Perturbations", IROS, 2017
    """

    name: str = 'arpl'

    def __init__(self,
                 save_dir: str,
                 env: [SimEnv, StateAugmentationWrapper],
                 subroutine: Algorithm,
                 policy: Policy,
                 expl_strat: StochasticActionExplStrat,
                 max_iter: int,
                 num_rollouts: int = None,
                 steps_num: int = None,
                 apply_dynamics_noise: bool = False,
                 dyn_eps: float = 0.01,
                 dyn_phi: float = 0.1,
                 halfspan: float = 0.25,
                 apply_proccess_noise: bool = False,
                 proc_eps: float = 0.01,
                 proc_phi: float = 0.05,
                 apply_observation_noise: bool = False,
                 obs_eps: float = 0.01,
                 obs_phi: float = 0.05,
                 torch_observation: bool = True,
                 base_seed: int = None,
                 num_sampler_envs: int = 4,
                 logger: StepLogger=None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment in which the agent should be trained
        :param subroutine: algorithm which performs the policy / value-function optimization
        :param policy: policy to be updated
        :param expl_strat: the exploration strategy
        :param max_iter: the maximum number of iterations
        :param num_rollouts: the number of rollouts to be performed for each update step
        :param steps_num: the number of steps to be performed for each update step
        :param apply_dynamics_noise: whether adversarially generated dynamics noise should be applied
        :param dyn_eps: the intensity of generated dynamics noise
        :param dyn_phi: the probability of applying dynamics noise
        :param halfspan: the halfspan of the uniform random distribution used to sample
        :param apply_proccess_noise: whether adversarially generated process noise should be applied
        :param proc_eps: the intensity of generated process noise
        :param proc_phi: the probability of applying process noise
        :param apply_observation_noise: whether adversarially generated observation noise should be applied
        :param obs_eps: the intensity of generated observation noise
        :param obs_phi: the probability of applying observation noise
        :param torch_observation: a function to provide a differentiable observation
        :param base_seed: the random seed
        :param num_sampler_envs: number of environments for parallel sampling
        :param logger: the logger
        """
        assert isinstance(subroutine, Algorithm)
        assert isinstance(max_iter, int) and max_iter > 0

        super().__init__(save_dir, max_iter, policy, logger)
        # Get the randomized environment (recommended to make it the most outer one in the chain)

        # Initialize adversarial wrappers
        if apply_dynamics_noise:
            assert isinstance(env, StateAugmentationWrapper)
            env = AdversarialDynamicsWrapper(env, self.policy, dyn_eps, dyn_phi, halfspan)
        if apply_proccess_noise:
            env = AdversarialStateWrapper(env, self.policy, proc_eps, proc_phi, torch_observation=torch_observation)
        if apply_observation_noise:
            env = AdversarialObservationWrapper(env, self.policy, obs_eps, obs_phi)

        self.num_rollouts = num_rollouts
        self.sampler = ParallelSampler(
            env,
            expl_strat,
            num_envs=num_sampler_envs,
            min_steps=steps_num,
            min_rollouts=num_rollouts,
            seed=base_seed
        )
        self._subroutine = subroutine

    def step(self, snapshot_mode: str, meta_info: dict = None):
        rollouts = self.sampler.sample()
        rets = [ro.undiscounted_return() for ro in rollouts]
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_std = np.std(rets)
        self.logger.add_value('num rollouts', len(rollouts))
        self.logger.add_value('avg rollout len', np.mean([ro.length for ro in rollouts]))
        self.logger.add_value('avg return', ret_avg)
        self.logger.add_value('median return', ret_med)
        self.logger.add_value('std return', ret_std)

        # Sub-routine
        self._subroutine.update(rollouts)
        self._subroutine.logger.record_step()
        self._subroutine.make_snapshot(snapshot_mode, ret_avg.item())
