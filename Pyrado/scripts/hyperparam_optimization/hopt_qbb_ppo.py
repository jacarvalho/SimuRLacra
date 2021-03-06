"""
Optimize the hyper-parameters of the Proximal Policy Optimization algorithm for the Quanser Ball-Balancer environment.
"""
import functools
import optuna
import os.path as osp
from optuna.pruners import MedianPruner

import pyrado
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.logger.experiment import save_list_of_dicts_to_yaml, setup_experiment
from pyrado.policies.fnn import FNNPolicy, FNN
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
    env = QBallBalancerSim(dt=1/250., max_steps=1500)
    env = ActNormWrapper(env)

    # Policy
    policy = FNNPolicy(
        spec=env.spec,
        hidden_sizes=trial.suggest_categorical('hidden_sizes_policy', [[16, 16], [32, 32], [64, 64]]),
        hidden_nonlin=fcn_from_str(trial.suggest_categorical('hidden_nonlin_policy', ['to_tanh', 'to_relu'])),
    )

    # Critic
    value_fcn = FNN(
        input_size=env.obs_space.flat_dim,
        output_size=1,
        hidden_sizes=trial.suggest_categorical('hidden_sizes_critic', [[16, 16], [32, 32], [64, 64]]),
        hidden_nonlin=fcn_from_str(trial.suggest_categorical('hidden_nonlin_critic', ['to_tanh', 'to_relu'])),
    )
    critic_hparam = dict(
        gamma=trial.suggest_uniform('gamma_critic', 0.99, 1.),
        lamda=trial.suggest_uniform('lamda_critic', 0.95, 1.),
        num_epoch=trial.suggest_int('num_epoch_critic', 1, 10),
        batch_size=100,
        lr=trial.suggest_loguniform('lr_critic', 1e-5, 1e-3),
        standardize_adv=trial.suggest_categorical('standardize_adv_critic', [True, False]),
        # max_grad_norm=5.,
        # lr_scheduler=scheduler.StepLR,
        # lr_scheduler_hparam=dict(step_size=10, gamma=0.9)
        # lr_scheduler=scheduler.ExponentialLR,
        # lr_scheduler_hparam=dict(gamma=0.99)
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        num_sampler_envs=1,  # parallelize via optuna n_jobs
        max_iter=500,
        min_steps=25*env.max_steps,
        num_epoch=trial.suggest_int('num_epoch_algo', 1, 10),
        eps_clip=trial.suggest_uniform('eps_clip_algo', 0.05, 0.2),
        batch_size=100,
        std_init=0.9,
        lr=trial.suggest_loguniform('lr_algo', 1e-5, 1e-3),
        # max_grad_norm=5.,
        # lr_scheduler=scheduler.StepLR,
        # lr_scheduler_hparam=dict(step_size=10, gamma=0.9)
        # lr_scheduler=scheduler.ExponentialLR,
        # lr_scheduler_hparam=dict(gamma=0.99)
    )
    algo = PPO(osp.join(ex_dir, f'trial_{trial.number}'), env, policy, critic, **algo_hparam)

    # Train without saving the results
    algo.train(snapshot_mode='latest', seed=seed)

    # Evaluate
    min_rollouts = 1000
    sampler = ParallelSampler(env, policy, num_envs=20, min_rollouts=min_rollouts)
    ros = sampler.sample()
    mean_ret = sum([r.undiscounted_return() for r in ros])/min_rollouts

    return mean_ret


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    ex_dir = setup_experiment('hyperparams', QBallBalancerSim.name, 'ppo_250Hz_actnorm', seed=args.seed)

    # Run hyper-parameter optimization
    name = f'{ex_dir.algo_name}_{ex_dir.add_info}'  # e.g. qbb_ppo_fnn_actnorm
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
