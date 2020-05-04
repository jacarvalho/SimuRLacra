import functools
import sys
from tqdm import tqdm

from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environments.base import Env
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers, DomainRandWrapperLive
from pyrado.environment_wrappers.utils import typed_env, remove_env
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


def _setup_env_policy(G, env, policy):
    G.env = env
    G.policy = policy


def _run_rollout_dp(G, domain_param, init_state=None):
    return rollout(G.env, G.policy, eval=True,  # render_mode=RenderMode(video=True),
                   reset_kwargs={'domain_param': domain_param, 'init_state': init_state})


def eval_domain_params(pool: SamplerPool, env: SimEnv, policy: Policy, params: list, init_state=None) -> list:
    """
    Evaluate a policy on a multidimensional grid of domain parameters.

    :param pool: parallel sampler
    :param env: environment to evaluate in
    :param policy: policy to evaluate
    :param params: multidimensional grid of domain parameters
    :param init_state: initial state of the environment which will be fixed if not set to None
    :return: list of rollouts
    """
    # Strip all domain randomization wrappers from the environment
    env = remove_all_dr_wrappers(env, verbose=True)

    pool.invoke_all(_setup_env_policy, env, policy)

    # Run with progress bar
    with tqdm(leave=False, file=sys.stdout, unit='rollouts', desc='Sampling') as pb:
        return pool.run_map(functools.partial(_run_rollout_dp, init_state=init_state), params, pb)


def _run_rollout_nom(G, init_state):
    return rollout(G.env, G.policy, eval=True, reset_kwargs={'init_state': init_state})


def eval_nominal_domain(pool: SamplerPool, env: SimEnv, policy: Policy, init_states: list) -> list:
    """
    Evaluate a policy using the nominal (set in the given environment) domain parameters.

    :param pool: parallel sampler
    :param env: environment to evaluate in
    :param policy: policy to evaluate
    :param init_states: initial states of the environment which will be fixed if not set to None
    :return: list of rollouts
    """
    # Strip all domain randomization wrappers from the environment
    env = remove_all_dr_wrappers(env, verbose=True)

    pool.invoke_all(_setup_env_policy, env, policy)

    # Run with progress bar
    with tqdm(leave=False, file=sys.stdout, unit='rollouts', desc='Sampling') as pb:
        return pool.run_map(_run_rollout_nom, init_states, pb)


def eval_randomized_domain(pool: SamplerPool,
                           env: SimEnv, randomizer: DomainRandomizer,
                           policy: Policy,
                           init_states: list) -> list:
    """
    Evaluate a policy in a randomized domain.

    :param pool: parallel sampler
    :param env: environment to evaluate in
    :param randomizer: randomizer used to sample random domain instances, inherited from `DomainRandomizer`
    :param policy: policy to evaluate
    :param init_states: initial states of the environment which will be fixed if not set to None
    :return: list of rollouts
    """
    # Randomize the environments
    env = DomainRandWrapperLive(env, randomizer)

    pool.invoke_all(_setup_env_policy, env, policy)

    # Run with progress bar
    with tqdm(leave=False, file=sys.stdout, unit='rollouts', desc='Sampling') as pb:
        return pool.run_map(_run_rollout_nom, init_states, pb)


def conditional_actnorm_wrapper(env: Env, ex_dirs: list, idx: int):
    """
    Wrap the environment with an action normalization wrapper if the simulated environment had one.

    :param env: environment to sample from
    :param ex_dirs: list of experiment directories that will be loaded
    :param idx: index of the current directory
    :return: modified environment
    """
    # Get the simulation environment
    env_sim, _, _ = load_experiment(ex_dirs[idx])

    if typed_env(env_sim, ActNormWrapper) is not None:
        env = ActNormWrapper(env)
        print_cbt(f'Added an action normalization wrapper to {idx + 1}-th evaluation policy.', 'y')
    else:
        env = remove_env(env, ActNormWrapper)
        print_cbt(f'Removed an action normalization wrapper to {idx + 1}-th evaluation policy.', 'y')
    return env
