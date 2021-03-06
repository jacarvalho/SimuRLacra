"""
Simulate (with animation) a rollout in an environment for all policies generated by Bayesian Domain Adaptation.
"""
import joblib
import numpy as np
import os
import os.path as osp
import torch as to
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import pyrado
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.logger.experiment import ask_for_experiment, load_dict_from_yaml
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt
from pyrado.utils.data_types import RenderMode

if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()
    if not osp.isdir(ex_dir):
        raise pyrado.PathErr(given=ex_dir)

    # Load the environment randomizer
    env_sim = joblib.load(osp.join(ex_dir, 'env_sim.pkl'))
    hparam = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))

    # Override the time step size if specified
    if args.dt is not None:
        env_sim.dt = args.dt

    # Crawl through the given directory and check how many init policies and candidates there are
    for root, dirs, files in os.walk(ex_dir):
        if args.load_all:
            found_policies = [p for p in files if p.endswith('_policy.pt')]
            found_cands = [c for c in files if c.endswith('_candidate.pt')]
        else:
            found_policies = [p for p in files if not p.startswith('init_') and p.endswith('_policy.pt')]
            found_cands = [c for c in files if not c.startswith('init_') and c.endswith('_candidate.pt')]
    if not len(found_policies) == len(found_cands):  # don't count the final policy
        raise pyrado.ValueErr(msg='Found a different number of initial policies than candidates!')

    # Sort (actually does not sort properly, e.g. 1, 10, 11, 2, 3 ...)
    found_policies.sort()
    found_cands.sort()

    # Plot the candidate values
    fig, ax = plt.subplots(1)
    for i in range(len(found_cands)):
        cand = to.load(osp.join(ex_dir, found_cands[i])).numpy()
        ax.scatter(np.arange(cand.size), cand, label='$\phi_{' + str(i) + '}$', c=f'C{i%10}', s=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('parameter value')
    ax.set_xlabel('parameter index')
    plt.legend()
    plt.show()

    # Simulate
    for i in range(len(found_policies)):
        # Load current
        policy = to.load(osp.join(ex_dir, found_policies[i]))
        cand = to.load(osp.join(ex_dir, found_cands[i]))

        # Set the domain randomizer given the hyper-parameters
        if isinstance(env_sim, MetaDomainRandWrapper):
            env_sim.adapt_randomizer(cand)
            print_cbt(f'Set the domain randomizer to\n{env_sim.randomizer}', 'c')
        else:
            raise pyrado.TypeErr(given=env_sim, expected_type=MetaDomainRandWrapper)

        done, state, param = False, None, None
        while not done:
            print_cbt(f'Simulating {found_policies[i]} with associated domain parameter distribution.', 'g')
            ro = rollout(env_sim, policy, render_mode=RenderMode(video=True), eval=True,
                         reset_kwargs=dict(domain_param=param, init_state=state))  # calls env.reset()
            print_domain_params(env_sim.domain_param)
            print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
            done, state, param = after_rollout_query(env_sim, policy, ro)
    pyrado.close_vpython()
