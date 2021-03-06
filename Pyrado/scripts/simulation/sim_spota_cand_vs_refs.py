import joblib
import os.path as osp
import pandas as pd
import torch as to

import pyrado
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.environments.sim_base import SimEnv
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer, DomainRandWrapperLive
from pyrado.environment_wrappers.utils import remove_env
from pyrado.logger.experiment import ask_for_experiment, load_dict_from_yaml
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


def sim_policy_fixed_env(env: SimEnv, policy: Policy, domain_param: [dict, list]):
    """
    Simulate (with animation) a rollout in a environment with fixed domain parameters.

    :param env: environment stack as it was used during training
    :param policy: policy to simulate
    :param domain_param: domain parameter set or a list of sets that specify the environment
    """
    # Remove wrappers that make the rollouts stochastic
    env = remove_env(env, GaussianObsNoiseWrapper)
    env = remove_env(env, DomainRandWrapperBuffer)
    env = remove_env(env, DomainRandWrapperLive)

    # Initialize
    done, state, i = False, None, 0
    if isinstance(domain_param, dict):
        param = domain_param
    elif isinstance(domain_param, list):
        param = domain_param[i]
    else:
        raise pyrado.TypeErr(given=domain_param, expected_type=[dict, list])

    while not done:
        ro = rollout(env, policy, reset_kwargs=dict(domain_param=param, init_state=state),
                     render_mode=RenderMode(video=True), eval=True)
        print_domain_params(env.domain_param)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, _ = after_rollout_query(env, policy, ro)

        if isinstance(domain_param, list):
            # Iterate over the list of domain parameter sets
            i = (i + 1) % len(domain_param)
            param = domain_param[i]


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    if args.iter < 0:
        # Select from the last iteration backward
        iter_sel = pd.read_csv(osp.join(ex_dir, 'progress.csv')).spota_iteration.max() + args.iter + 1
    else:
        iter_sel = args.iter

    # Load the environment and the policies
    env = joblib.load(osp.join(ex_dir, 'init_env.pkl'))
    hparams = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))
    policy_cand = to.load(osp.join(ex_dir, f'iter_{iter_sel}_policy_cand.pt'))
    policy_refs = [to.load(osp.join(ex_dir, f'iter_{iter_sel}_policy_ref_{i}.pt'))
                   for i in range(hparams['SPOTA']['nG'])]

    # Override the time step size if specified
    if args.dt is not None:
        env.dt = args.dt

    # Candidate
    # Load the domain parameters of the candidate solution
    domain_param_cand = joblib.load(osp.join(ex_dir, f'iter_{iter_sel}_env_params_cand.pkl'))

    print_cbt('Started candidate policy.', 'c', bright=True)
    sim_policy_fixed_env(env, policy_cand, domain_param_cand)
    print_cbt('Finished candidate policy.', 'c', bright=True)

    # References
    for i in range(hparams['SPOTA']['nG']):
        # Load the domain parameters of the current reference solution
        domain_param_ref = joblib.load(osp.join(ex_dir, f'iter_{iter_sel}_env_params_ref_{i}.pkl'))

        print_cbt(f'Started reference policy {i}.', 'c', bright=True)
        sim_policy_fixed_env(env, policy_refs[i], domain_param_ref)
        print_cbt(f'Finished reference policy {i}.', 'c', bright=True)
    pyrado.close_vpython()
