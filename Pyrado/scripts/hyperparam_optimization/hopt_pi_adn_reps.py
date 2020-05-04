"""
Optimize the hyper-parameters of the PRelative Entropy Policy Search algorithm for the Planar Insert environment.
"""
import functools
import optuna
import os.path as osp
import torch as to
from optuna.pruners import MedianPruner

import pyrado
from pyrado.algorithms.reps import REPS
from pyrado.environments.rcspysim.planar_insert import PlanarInsertSim
from pyrado.logger.experiment import save_list_of_dicts_to_yaml, setup_experiment
from pyrado.policies.adn import ADNPolicy, pd_cubic, pd_capacity_21
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
    env = PlanarInsertSim(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gPlanarInsert6Link.xml',
        dt=1/100.,
        max_steps=700,
        max_dist_force=None,
        checkJointLimits=False,
    )

    # Policy
    policy_hparam = dict(
        tau_init=trial.suggest_uniform('tau_init', 1., 5.),
        tau_learnable=True,
        output_nonlin=to.sigmoid,
        potentials_dyn_fcn=trial.suggest_categorical('potentials_dyn_fcn', [pd_cubic, pd_capacity_21]),
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        num_sampler_envs=1,  # parallelize via optuna n_jobs
        max_iter=200,
        eps=trial.suggest_uniform('eps', 0.05, 0.2),
        gamma=trial.suggest_uniform('gamma', 0.99, 1.),
        pop_size=trial.suggest_categorical('pop_size', [50, 100, 200]),
        num_rollouts=trial.suggest_categorical('num_rollouts', [5, 10, 20]),
        expl_std_init=trial.suggest_categorical('expl_std_init', [0.5, 1., 2.]),
        expl_std_min=trial.suggest_categorical('expl_std_min', [0.02, 0.05, 0.1]),
        num_epoch_dual=1000,
        lr_dual=1e-4,
        use_map=True,
    )
    algo = REPS(ex_dir, env, policy, **algo_hparam)

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

    ex_dir = setup_experiment('hyperparams', PlanarInsertSim.name, 'reps_adn_100Hz', seed=args.seed)

    # Run hyper-parameter optimization
    name = f'{ex_dir.algo_name}_{ex_dir.add_info}'  # e.g. pi_adn_reps_100Hz
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
