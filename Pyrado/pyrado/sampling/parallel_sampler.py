import sys
import pickle
from init_args_serializer import Serializable
from tqdm import tqdm
from typing import List
import torch.multiprocessing as mp


from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sampler import SamplerBase


def _ps_init(G, env, policy, bernoulli_reset: bool):
    """ Store pickled (and thus copied) environment as well as policy. """
    G.env = pickle.loads(env)
    G.policy = pickle.loads(policy)
    G.bernoulli_reset = pickle.loads(bernoulli_reset)


def _ps_update_policy(G, state):
    """ Update policy state_dict. """
    G.policy.load_state_dict(state)


def _ps_sample_one(G):
    """
    Sample one rollout and return step count if counting steps, rollout count (1) otherwise.
    This function is used when a minimum number of steps was given.
    """
    ro = rollout(G.env, G.policy, bernoulli_reset=G.bernoulli_reset)
    return ro, len(ro)


def _ps_run_one(G, num):
    """
    Sample one rollout.
    This function is used when a minimum number of rollouts was given.
    """
    return rollout(G.env, G.policy, bernoulli_reset=G.bernoulli_reset)


class ParallelSampler(SamplerBase, Serializable):
    """ Class for sampling from multiple environments in parallel """

    def __init__(self,
                 env,
                 policy,
                 num_envs: int,
                 *,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 bernoulli_reset: bool = None,
                 seed: int = None):
        """
        Constructor

        :param env: environment to sample from
        :param policy: policy to act in the environment (can also be an exploration strategy)
        :param num_envs: number of parallel samplers
        :param min_rollouts: minimum number of complete rollouts to sample.
        :param min_steps: minimum total number of steps to sample.
        :param bernoulli_reset: probability for resetting after the current time step
        :param seed: Seed to use. Every subprocess is seeded with seed+thread_number
        """
        Serializable._init(self, locals())
        super().__init__(min_rollouts=min_rollouts, min_steps=min_steps)

        self.env = env
        self.policy = policy
        self.bernoulli_reset = bernoulli_reset

        # Set method to spawn if using cuda
        if self.policy.device == 'cuda':
            mp.set_start_method('spawn', force=True)

        # Create parallel pool. We use one thread per env because it's easier.
        self.pool = SamplerPool(num_envs)

        if seed is not None:
            self.pool.set_seed(seed)

        # Distribute environments. We use pickle to make sure a copy is created for n_envs=1
        self.pool.invoke_all(
            _ps_init, pickle.dumps(self.env), pickle.dumps(self.policy), pickle.dumps(self.bernoulli_reset)
        )

    def reinit(self, env=None, policy=None, bernoulli_reset: bool = None):
        """ Re-initialize the sampler. """
        # Update env and policy if passed
        if env is not None:
            self.env = env
        if policy is not None:
            self.policy = policy
        if bernoulli_reset is not None:
            self.bernoulli_reset = bernoulli_reset

        # Always broadcast to workers
        self.pool.invoke_all(
            _ps_init, pickle.dumps(self.env), pickle.dumps(self.policy), pickle.dumps(self.bernoulli_reset)
        )

    def sample(self) -> List[StepSequence]:
        """ Do the sampling according to the previously given environment, policy, and number of steps/rollouts. """
        # Update policy's state
        self.pool.invoke_all(_ps_update_policy, self.policy.state_dict())

        # Collect samples
        with tqdm(leave=False, file=sys.stdout, desc='Sampling',
                  unit='steps' if self.min_steps is not None else 'rollouts') as pb:

            if self.min_steps is None:
                # Only minimum number of rollouts given, thus use run_map
                return self.pool.run_map(_ps_run_one, range(self.min_rollouts), pb)
            else:
                # Minimum number of steps given, thus use run_collect (automatically handles min_runs=None)
                return self.pool.run_collect(
                    self.min_steps,
                    _ps_sample_one,
                    collect_progressbar=pb,
                    min_runs=self.min_rollouts
                )[0]
