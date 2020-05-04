"""
Optimize the hyper-parameters of the Proximal Policy Optimization algorithm for the Quanser Ball-Balancer environment.
"""
import functools
import optuna
import os.path as osp
from optuna.pruners import MedianPruner

import pyrado
from pyrado.algorithms.nes import NES
from pyrado.environments.rcspysim.box_shelving import BoxShelvingVelMPsSim
from pyrado.logger.experiment import save_list_of_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.adn import ADNPolicy
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
    env_hparams = dict(
        physicsEngine=trial.suggest_categorical('physicsEngine', ['Bullet']),  # Bullet or Vortex
        graphFileName=trial.suggest_categorical('graphFileName', ['gBoxShelving_posCtrl.xml', 'gBoxShelving_trqCtrl.xml']),
        dt=trial.suggest_categorical('dt', [1/100.]),
        max_steps=trial.suggest_categorical('max_steps', [2000]),
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        collisionConfig={'file': 'collisionModel.xml'},
        checkJointLimits=trial.suggest_categorical('checkJointLimits', [True]),
        collisionAvoidanceIK=trial.suggest_categorical('collisionAvoidanceIK', [True]),
        observeCollisionCost=trial.suggest_categorical('observeCollisionCost', [True]),
        observePredictedCollisionCost=trial.suggest_categorical('observePredictedCollisionCost', [True, False]),
        observeManipulabilityIndex=trial.suggest_categorical('observeManipulabilityIndex', [True, False]),
        observeDynamicalSystemDiscrepancy=trial.suggest_categorical('observeDynamicalSystemDiscrepancy', [True, False]),
        observeTaskSpaceDiscrepancy=trial.suggest_categorical('observeTaskSpaceDiscrepancy', [True]),
        observeDSGoalDistance=trial.suggest_categorical('observeDSGoalDistance', [True, False]),
    )
    env = BoxShelvingVelMPsSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        tau_init=trial.suggest_uniform('tau_init', 2., 5.),
        tau_learnable=trial.suggest_categorical('tau_learnable', [False]),
        kappa_learnable=trial.suggest_categorical('kappa_learnable', [True]),
        capacity_learnable=trial.suggest_categorical('capacity_learnable', [True]),
        output_nonlin=fcn_from_str(trial.suggest_categorical('output_nonlin', ['to_sigmoid'])),
        potentials_dyn_fcn=fcn_from_str(trial.suggest_categorical(
            'hidden_nonlin_critic', ['pd_cubic', 'pd_capacity_21', 'pd_capacity_21_abs'])
        ),
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        num_sampler_envs=1,  # parallelize via optuna n_jobs
        max_iter=1500,
        pop_size=trial.suggest_categorical('pop_size', [None]),
        num_rollouts=trial.suggest_int('num_rollouts', 1, 10),
        eta_mean=trial.suggest_uniform('eta_mean', 0.5, 5.),
        eta_std=trial.suggest_categorical('eta_std', [None]),
        expl_std_init=trial.suggest_uniform('expl_std_init', 0.2, 1.5),
        symm_sampling=trial.suggest_categorical('symm_sampling', [True, False]),
        transform_returns=trial.suggest_categorical('transform_returns', [True]),
    )
    csv_logger = create_csv_step_logger(osp.join(ex_dir, f'trial_{trial.number}'))
    algo = NES(osp.join(ex_dir, f'trial_{trial.number}'), env, policy, **algo_hparam, logger=csv_logger)

    # Train without saving the results
    algo.train(snapshot_mode='latest', seed=seed)

    # Evaluate
    min_rollouts = 1000
    sampler = ParallelSampler(env, policy, num_envs=1, min_rollouts=min_rollouts)   # parallelize via optuna n_jobs
    ros = sampler.sample()
    mean_ret = sum([r.undiscounted_return() for r in ros])/min_rollouts

    return mean_ret


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    ex_dir = setup_experiment('hyperparams', BoxShelvingVelMPsSim.name, 'nes_adn', seed=args.seed)

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
