"""
Script to evaluate multiple policies in one environment using the nominal domain parameters.
"""
import os.path as osp
import pandas as pd
import pprint

import pyrado
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim, QCartPoleStabSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.sampling.parallel_evaluation import eval_nominal_domain, conditional_actnorm_wrapper
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import dict_arraylike_to_float
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    if args.max_steps == pyrado.inf:
        args.max_steps = 2500
        print_cbt(f'Set maximum number of time steps to {args.max_steps}', 'y')

    if args.env_name == QBallBalancerSim.name:
        # Create the environment for evaluating
        env = QBallBalancerSim(dt=args.dt, max_steps=args.max_steps)

        # Get the experiments' directories to load from
        prefixes = [
                osp.join(pyrado.EXP_DIR, 'FILL_IN', 'FILL_IN'),
        ]
        exp_names = [
                '',
        ]
        exp_labels = [
                '',
        ]

    elif args.env_name in [QCartPoleStabSim.name, QCartPoleSwingUpSim.name]:
        # Create the environment for evaluating
        if args.env_name == QCartPoleSwingUpSim.name:
            env = QCartPoleSwingUpSim(dt=args.dt, max_steps=args.max_steps)
        else:
            env = QCartPoleStabSim(dt=args.dt, max_steps=args.max_steps)

        # Get the experiments' directories to load from
        prefixes = [
                osp.join(pyrado.EXP_DIR, 'FILL_IN', 'FILL_IN'),
        ]
        exp_names = [
                '',
        ]
        exp_labels = [
                '',
        ]

    else:
        raise pyrado.ValueErr(given=args.env_name, eq_constraint=f'{QBallBalancerSim.name}, {QCartPoleStabSim.name},'
                                                                 f'or {QCartPoleSwingUpSim.name}')

    if not (len(prefixes) == len(exp_names) and len(prefixes) == len(exp_labels)):
        raise pyrado.ShapeErr(msg=f'The lengths of prefixes, exp_names, and exp_labels must be equal, but they'
                                  f' are {len(prefixes)}, {len(exp_names)}, and {len(exp_labels)}!')

    # Load the policies
    ex_dirs = [osp.join(p, e) for p, e in zip(prefixes, exp_names)]
    policies = []
    for ex_dir in ex_dirs:
        _, policy, _ = load_experiment(ex_dir)
        policies.append(policy)

    # Fix initial state (set to None if it should not be fixed)
    init_state_list = [None] * args.num_ro_per_config

    # Crate empty data frame
    df = pd.DataFrame(columns=['policy', 'ret', 'len'])

    # Evaluate all policies
    for i, policy in enumerate(policies):
        # Create a new sampler pool for every policy to synchronize the random seeds i.e. init states
        pool = SamplerPool(args.num_envs)

        # Seed the sampler
        if args.seed is not None:
            pool.set_seed(args.seed)
            print_cbt(f'Set seed to {args.seed}', 'y')
        else:
            print_cbt('No seed was set', 'r', bright=True)

        # Add an action normalization wrapper if the policy was trained with one
        env = conditional_actnorm_wrapper(env, ex_dirs, i)

        # Sample rollouts
        ros = eval_nominal_domain(pool, env, policy, init_state_list)

        # Compute results metrics
        rets = [ro.undiscounted_return() for ro in ros]
        lengths = [float(ro.length) for ro in ros]  # int values are not numeric in pandas
        df = df.append(pd.DataFrame(dict(policy=exp_labels[i], ret=rets, len=lengths)), ignore_index=True)

    metrics = dict(
            avg_len=df.groupby('policy').mean()['len'].to_dict(),
            avg_ret=df.groupby('policy').mean()['ret'].to_dict(),
            median_ret=df.groupby('policy').median()['ret'].to_dict(),
            min_ret=df.groupby('policy').min()['ret'].to_dict(),
            max_ret=df.groupby('policy').max()['ret'].to_dict(),
            std_ret=df.groupby('policy').std()['ret'].to_dict()
    )
    pprint.pprint(metrics)

    # Create sub-folder and save
    save_dir = setup_experiment('multiple_policies', args.env_name, 'nominal', base_dir=pyrado.EVAL_DIR)

    save_list_of_dicts_to_yaml(
            [{'ex_dirs': ex_dirs},
             {'num_rpp': args.num_ro_per_config, 'seed': args.seed},
             dict_arraylike_to_float(metrics)],
            save_dir, file_name='summary'
    )
    df.to_pickle(osp.join(save_dir, 'df_nom_mp.pkl'))
