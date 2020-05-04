"""
Train an agent to solve the randomized Quanser Qube environment using Proximal Policy Optimization.
"""
import torch as to

from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.domain_randomization.default_randomizers import get_uniform_masses_lengths_randomizer_qq
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import RNNPolicy, LSTMPolicy, GRUPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, f'udr-{PPO.name}', 'fnn_actnorm_dr-masses-lengths', seed=2)
    # ex_dir = setup_experiment(QQubeSim.name, f'udr-{PPO.name}', 'gru_actnorm_dr-masses-lengths', seed=101)

    # Environment
    env_hparams = dict(dt=1/250., max_steps=1500)
    env = QQubeSim(**env_hparams)
    env = ActNormWrapper(env)
    env = DomainRandWrapperLive(env, get_uniform_masses_lengths_randomizer_qq(frac_halfspan=10.))

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu)  # FNN
    # policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = RNNPolicy(spec=env.spec, **policy_hparam)
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    # policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)  # FNN
    # value_fcn_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.9995,
        lamda=0.95,
        num_epoch=5,
        batch_size=100,
        # max_grad_norm=5,
        lr=5e-4,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
        min_steps=30*env.max_steps,
        num_sampler_envs=6,
        num_epoch=5,
        eps_clip=0.15,
        batch_size=100,
        std_init=1.0,
        # max_grad_norm=5,
        lr=2e-4,
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    sm = 'best'
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed, snapshot_mode=sm),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode=sm, seed=ex_dir.seed)
