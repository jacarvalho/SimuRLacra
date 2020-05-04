"""
Optimize the hyper-parameters of the Deep Q-Leaning algorithm for the Ball-on-Beam environment.
"""
import functools
import optuna
import os.path as osp
from optuna.pruners import MedianPruner

import pyrado
from pyrado.algorithms.dql import DQL
from pyrado.environments.pysim.ball_on_beam import BallOnBeamDiscSim
from pyrado.logger.experiment import save_list_of_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.fnn import DiscrActQValFNNPolicy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import fcn_from_str


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
    env_hparams = dict(dt=1/100., max_steps=500)
    env = BallOnBeamDiscSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        hidden_sizes=trial.suggest_categorical('hidden_sizes_policy', [[16, 16], [32, 32], [64, 64]]),
        hidden_nonlin=fcn_from_str(trial.suggest_categorical('hidden_nonlin_policy', ['to_tanh', 'to_relu'])),
    )
    policy = DiscrActQValFNNPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    ms = trial.suggest_categorical('min_steps_algo', [1, 10, env.max_steps, 10*env.max_steps])
    algo_hparam = dict(
        num_sampler_envs=1,  # parallelize via optuna n_jobs
        max_iter=int(1e5/ms),
        min_steps=ms,
        memory_size=trial.suggest_categorical('memory_size_algo', [10*env.max_steps, 100*env.max_steps, 1000*env.max_steps]),
        eps_init=trial.suggest_uniform('eps_init_algo', 0.1, 0.5),
        eps_schedule_gamma=trial.suggest_uniform('eps_schedule_gamma_algo', 0.995, 1.),
        gamma=0.998,
        target_update_intvl=trial.suggest_categorical('target_update_intvl_algo', [1, 5, 10, 20, 50]),
        num_batch_updates=trial.suggest_categorical('num_batch_updates_algo', [1, 5, 10]),
        batch_size=trial.suggest_categorical('batch_size_algo', [100, 200]),
        max_grad_norm=trial.suggest_categorical('max_grad_norm_algo', [0.5]),
        lr=trial.suggest_loguniform('lr_algo', 1e-5, 1e-3),
    )
    csv_logger = create_csv_step_logger(osp.join(ex_dir, f'trial_{trial.number}'))
    algo = DQL(ex_dir, env, policy, **algo_hparam, logger=csv_logger)

    # Train without saving the results
    algo.train(snapshot_mode='best', seed=seed)

    # Evaluate
    min_rollouts = 1000
    sampler = ParallelSampler(env, policy, num_envs=1, min_rollouts=min_rollouts)  # parallelize via optuna n_jobs
    ros = sampler.sample()
    mean_ret = sum([r.undiscounted_return() for r in ros])/min_rollouts

    return mean_ret


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    ex_dir = setup_experiment('hyperparams', BallOnBeamDiscSim.name, 'dql_fnn', seed=args.seed)

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
