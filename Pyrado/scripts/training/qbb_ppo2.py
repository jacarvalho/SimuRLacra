"""
Train an agent to solve the Quanser Ball-Balancer environment using Proximal Policy Optimization.
"""
import torch as to
from numpy import pi
from torch.optim import lr_scheduler as scheduler

from pyrado.algorithms.ppo import PPO2
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import LSTMPolicy, GRUPolicy
from pyrado.utils.data_types import EnvSpec

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QBallBalancerSim.name, PPO2.name, f'{FNNPolicy.name}_obsnoise_actnorm', seed=1001)

    # Environment
    env_hparams = dict(dt=1/500., max_steps=2500)
    env = QBallBalancerSim(**env_hparams)
    env = GaussianObsNoiseWrapper(env, noise_std=[1/180*pi, 1/180*pi, 0.0025, 0.0025,  # [rad, rad, m, m, ...
                                                  2/180*pi, 2/180*pi, 0.05, 0.05])  # ... rad/s, rad/s, m/s, m/s]
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)  # FNN
    # policy_hparam = dict(hidden_size=64, num_recurrent_layers=1)  # LSTM & GRU
    policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    # policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    # value_fcn_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.999,
        lamda=0.98,
        num_epoch=3,
        batch_size=100,
        lr=5e-4,
        max_grad_norm=5.,
        # lr_scheduler=scheduler.StepLR,
        # lr_scheduler_hparam=dict(step_size=10, gamma=0.9)
        # lr_scheduler=scheduler.ExponentialLR,
        # lr_scheduler_hparam=dict(gamma=0.99)
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
        min_steps=30*env.max_steps,
        num_sampler_envs=4,
        num_epoch=3,
        value_fcn_coeff=0.7,
        entropy_coeff=1e-4,
        eps_clip=0.1,
        batch_size=100,
        std_init=0.8,
        lr=2e-4,
        max_grad_norm=5.,
        # lr_scheduler=scheduler.StepLR,
        # lr_scheduler_hparam=dict(step_size=10, gamma=0.9)
        # lr_scheduler=scheduler.ExponentialLR,
        # lr_scheduler_hparam=dict(gamma=0.99)
    )
    algo = PPO2(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
