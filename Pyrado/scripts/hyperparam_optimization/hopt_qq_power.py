"""
Optimize the hyper-parameters of the Proximal Policy Optimization algorithm for the Quanser Ball-Balancer environment.
"""
import functools
import optuna
import os.path as osp
from optuna.pruners import MedianPruner

import pyrado
from pyrado.algorithms.power import PoWER
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.logger.experiment import save_list_of_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.features import FeatureStack, identity_feat, sign_feat, abs_feat, squared_feat, \
    bell_feat, MultFeat
from pyrado.policies.linear import LinearPolicy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.utils.argparser import get_argparser


def train_and_eval(trial: optuna.Trial, ex_dir: str, seed: [int, None]):
    """
    Objective function for the Optuna `Study` to maximize.

    .. note::
        Optuna expects only the `trial` argument, thus we use `functools.partial` to sneak in custom arguments.

    :param trial: Optuna Trial object for hyper-parameter optimization
    :param ex_dir: experiment's directory, i.e. the parent directory for all trials in this study
    :param seed: seed value for the random number generators, pass `None` for no seeding
    :return: objective function value
    """
    # Synchronize seeds between Optuna trials
    pyrado.set_seed(seed)

    # Environment
    env_hparams = dict(dt=1/250., max_steps=1500)
    env = QQubeSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([identity_feat, sign_feat, abs_feat, squared_feat, bell_feat, MultFeat([4, 5])])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        num_sampler_envs=1,  # parallelize via optuna n_jobs
        max_iter=150,
        pop_size=trial.suggest_categorical('pop_size', [100, 150, 200, 250]),
        num_rollouts=trial.suggest_categorical('num_rollouts', [4, 6, 8, 10, 12]),
        num_is_samples=trial.suggest_categorical('num_is_samples', [50, 100, 150, 200]),
        expl_std_init=trial.suggest_uniform('expl_std_init', 0.2, 1.5),
        expl_std_min=0.02,
        symm_sampling=trial.suggest_categorical('symm_sampling', [True, False]),
    )
    csv_logger = create_csv_step_logger(osp.join(ex_dir, f'trial_{trial.number}'))
    algo = PoWER(osp.join(ex_dir, f'trial_{trial.number}'), env, policy, **algo_hparam, logger=csv_logger)

    # Train without saving the results
    algo.train(snapshot_mode='latest', seed=seed)

    # Evaluate
    min_rollouts = 1000
    sampler = ParallelSampler(env, policy, num_envs=1, min_rollouts=min_rollouts)  # parallelize via optuna n_jobs
    ros = sampler.sample()
    mean_ret = sum([r.undiscounted_return() for r in ros])/min_rollouts

    return mean_ret


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    ex_dir = setup_experiment('hyperparams', QQubeSim.name, 'power_lin_actnorm', seed=args.seed)

    # Run hyper-parameter optimization
    name = f'{ex_dir.algo_name}_{ex_dir.add_info}'  # e.g. qq_ppo_fnn_actnorm
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:////{osp.join(pyrado.TEMP_DIR, ex_dir, f'{name}.db')}",
        direction='maximize',
        pruner=MedianPruner(),
        load_if_exists=True
    )
    study.optimize(functools.partial(train_and_eval, ex_dir=ex_dir, seed=args.seed), n_trials=100, n_jobs=6)

    # Save the best hyper-parameters
    save_list_of_dicts_to_yaml([study.best_params, dict(seed=args.seed)], ex_dir, 'best_hyperparams')
