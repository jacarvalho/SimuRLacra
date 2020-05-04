import itertools
import numpy as np
import pickle
import sys
import torch as to
from tqdm import tqdm
from typing import Sequence, List, NamedTuple

import pyrado
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper, DomainRandWrapperBuffer, \
    remove_all_dr_wrappers
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.step_sequence import StepSequence
from torch.nn.utils.convert_parameters import vector_to_parameters
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.rollout import rollout
from pyrado.utils.properties import cached_property
from pyrado.environment_wrappers.utils import typed_env, attr_env_get, inner_env


class ParameterSample(NamedTuple):
    """ Stores policy parameters and associated rollouts. """
    params: to.Tensor
    rollouts: List[StepSequence]

    @property
    def mean_undiscounted_return(self):
        return np.mean([r.undiscounted_return() for r in self.rollouts])

    @property
    def num_rollouts(self):
        return len(self.rollouts)


class ParameterSamplingResult(Sequence[ParameterSample]):
    """
    Result of a parameter exploration sampling run.
    On one hand, this is a list of ParameterSamples.
    On the other hand, this allows to query combined tensors of parameters and mean returns.
    """

    def __init__(self, samples: List[ParameterSample]):
        """
        Constructor

        :param samples: list of parameter samples
        """
        self._samples = samples

    def __getitem__(self, idx):
        # Get from samples
        res = self._samples[idx]
        if not isinstance(res, ParameterSample):
            # Was a slice, return a wrapped slice
            return ParameterSamplingResult(res)
        # Single item, return it
        return res

    def __len__(self):
        return len(self._samples)

    @cached_property
    def parameters(self) -> to.Tensor:
        """ Get all policy parameters as NxP matrix, where N is the number of samples and P is the policy param dim. """
        return to.stack([s.params for s in self._samples])

    @cached_property
    def mean_returns(self) -> np.ndarray:
        """ Get all parameter sample means return as a N-dim vector, where N is the number of samples. """
        return np.array([s.mean_undiscounted_return for s in self._samples])

    @cached_property
    def rollouts(self) -> list:
        """ Get all rollouts for all samples, i.e. a list of pop_size items, each a list of nom_rollouts rollouts. """
        return [s.rollouts for s in self._samples]

    @cached_property
    def num_rollouts(self) -> int:
        """ Get the total number of rollouts for all samples. """
        return int(np.sum([s.num_rollouts for s in self._samples]))


def _pes_init(G, env, policy):
    # Store pickled (and thus copied) env/policy
    G.env = pickle.loads(env)
    G.policy = pickle.loads(policy)


def _pes_sample_one(G, param):
    # Sample one rollout with param
    pol_param, dom_param, init_state = param

    vector_to_parameters(pol_param, G.policy.parameters())
    return rollout(G.env, G.policy, reset_kwargs={
        "init_state": init_state,
        "domain_param": dom_param,
    })


class ParameterExplorationSampler:
    """ Parallel sampler for parameter exploration """

    def __init__(self,
                 env: Env,
                 policy: Policy,
                 num_envs: int,
                 num_rollouts_per_param: int,
                 seed: int = None):
        """
        Constructor

        :param env: environment to sample from
        :param policy: policy used for sampling
        :param num_envs: number of parallel samplers
        :param num_rollouts_per_param:
        :param seed:
        """
        # Check environment for domain randomization wrappers (stops after finding the outermost)
        self._dp_wrapper = typed_env(env, DomainRandWrapper)
        if self._dp_wrapper is not None:
            assert isinstance(inner_env(env), SimEnv)
            # Remove it from env chain
            env = remove_all_dr_wrappers(env)

        self.env, self.policy = env, policy
        self.num_rollouts_per_param = num_rollouts_per_param

        # Create parallel pool. We use one thread per env because it's easier.
        self.pool = SamplerPool(num_envs)

        if seed is not None:
            self.pool.set_seed(seed)

        # Distribute environments. We use pickle to make sure a copy is created for n_envs=1
        self.pool.invoke_all(_pes_init, pickle.dumps(self.env), pickle.dumps(self.policy))

    def _sample_domain_params(self) -> [list, dict]:
        # Sample domain params from wrapper
        if self._dp_wrapper is None:
            # No params
            return [None]*self.num_rollouts_per_param
        elif isinstance(self._dp_wrapper, DomainRandWrapperBuffer) and self._dp_wrapper.buffer is not None:
            # Use buffered param sets
            idcs = np.random.randint(0, len(self._dp_wrapper.buffer), size=self.num_rollouts_per_param)
            return [self._dp_wrapper.buffer[i] for i in idcs]

        # Sample new ones
        rand = self._dp_wrapper.randomizer
        rand.randomize(self.num_rollouts_per_param)
        return rand.get_params(-1, format='list', dtype='numpy')

    def _sample_one_init_state(self, domain_param: dict) -> [np.ndarray, None]:
        """
        Sample an init state for the given domain parameter set(s).
        For some environments, the initial state space depends on the domain parameters, so we need to set them before
        sampling it. We can just reset `self.env` here safely though, since it's not used for anything else.

        :param domain_param: domain parameters to set
        :return: initial state, `None` if no initial state space is defined
        """
        self.env.reset(domain_param=domain_param)
        ispace = attr_env_get(self.env, 'init_space')
        if ispace is not None:
            return ispace.sample_uniform()
        else:
            # No init space, no init state
            return None

    def sample(self, param_sets: to.Tensor) -> ParameterSamplingResult:
        """ Sample rollouts for a given set of parameters. """
        # Sample domain params for each rollout
        domain_params = self._sample_domain_params()

        if isinstance(domain_params, dict):
            # There is only one domain parameter set (i.e. one init state)
            init_states = [self._sample_one_init_state(domain_params)]
            domain_params = [domain_params]  # cast to list of dict to make iterable like the next case
        elif isinstance(domain_params, list):
            # There are more than one domain parameter set (i.e. multiple init states)
            init_states = [self._sample_one_init_state(dp) for dp in domain_params]
        else:
            pyrado.TypeErr(given=domain_params, expected_type=[list, dict])

        # Explode parameter list for rollouts per param
        all_params = [(p, *r)
                      for p in param_sets
                      for r in zip(domain_params, init_states)]

        # Sample rollouts in parallel
        with tqdm(leave=False, file=sys.stdout, desc='Sampling',
                  unit='rollouts') as pb:
            all_ros = self.pool.run_map(_pes_sample_one, all_params, pb)

        # Group rollouts by parameters
        ros_iter = iter(all_ros)
        return ParameterSamplingResult([
            ParameterSample(params=p, rollouts=list(itertools.islice(ros_iter, self.num_rollouts_per_param)))
            for p in param_sets
        ])
