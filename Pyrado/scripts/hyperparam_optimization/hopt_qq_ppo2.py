"""
Optimize the hyper-parameters of the Proximal Policy Optimization algorithm for the Quanser Ball-Balancer environment.
"""
import functools
import optuna
import os.path as osp
from optuna.pruners import MedianPruner

import pyrado
from pyrado.algorithms.ppo import PPO2
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.logger.experiment import save_list_of_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import GRUPolicy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
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
    env_hparams = dict(dt=1/250., max_steps=1500)
    env = QQubeSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        hidden_sizes=trial.suggest_categorical('hidden_sizes_policy', [[32], [64], [16, 16], [32, 32], [64, 64]]),
        hidden_nonlin=fcn_from_str(trial.suggest_categorical('hidden_nonlin_policy', ['to_tanh', 'to_relu'])),
    )  # FNN
    # policy_hparam = dict(
    #     hidden_size=trial.suggest_categorical('hidden_size_policy', [16, 32, 64]),
    #     num_recurrent_layers=trial.suggest_categorical('num_recurrent_layers_policy', [1, 2]),
    # )  # LSTM & GRU
    policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(
        hidden_sizes=trial.suggest_categorical('hidden_sizes_critic', [[32], [64], [16, 16], [32, 32], [64, 64]]),
        hidden_nonlin=fcn_from_str(trial.suggest_categorical('hidden_nonlin_critic', ['to_tanh', 'to_relu'])),
    )
    # value_fcn_hparam = dict(
    #     hidden_size=trial.suggest_categorical('hidden_size_critic', [16, 32, 64]),
    #     num_recurrent_layers=trial.suggest_categorical('num_recurrent_layers_critic', [1, 2]),
    # )  # LSTM & GRU
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=trial.suggest_uniform('gamma_critic', 0.98, 1.),
        lamda=trial.suggest_uniform('lamda_critic', 0.90, 1.),
        batch_size=150,
        standardize_adv=True,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        num_sampler_envs=1,  # parallelize via optuna n_jobs
        max_iter=300,
        min_steps=trial.suggest_int('num_rollouts_algo', 5, 30)*env.max_steps,
        num_epoch=trial.suggest_int('num_epoch_algo', 1, 20),
        eps_clip=trial.suggest_uniform('eps_clip_algo', 0.05, 0.2),
        batch_size=150,
        std_init=trial.suggest_uniform('std_init', 0.5, 1.0),
        lr=trial.suggest_loguniform('lr_algo', 1e-5, 1e-3),
        value_fcn_coeff=trial.suggest_uniform('value_fcn_coeff_algo', 0.2, 0.8),
        entropy_coeff=trial.suggest_loguniform('entropy_coeff_algo', 1e-6, 1e-3),
        max_grad_norm=trial.suggest_categorical('max_grad_norm_algo', [1., 2., 5., None]),
    )
    csv_logger = create_csv_step_logger(osp.join(ex_dir, f'trial_{trial.number}'))
    algo = PPO2(osp.join(ex_dir, f'trial_{trial.number}'), env, policy, critic, **algo_hparam, logger=csv_logger)

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

    ex_dir = setup_experiment('hyperparams', QQubeSim.name, 'ppo2_fnn_actnorm', seed=args.seed)
    # ex_dir = setup_experiment('hyperparams', QQubeSim.name, 'ppo2_gru_actnorm', seed=args.seed)

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
