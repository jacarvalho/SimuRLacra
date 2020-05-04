"""
Train an agent to solve the inverted pendulum environment using Proximal Policy Optimization.
"""
import torch as to

from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.algorithms.ppo import PPO2
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import GRUPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    # ex_dir = setup_experiment(PendulumSim.name, PPO2.name, 'fnn', seed=1001)
    ex_dir = setup_experiment(PendulumSim.name, PPO2.name, 'gru', seed=1001)

    # Environment
    env_hparams = dict(dt=1/50., max_steps=800)
    env = PendulumSim(**env_hparams)

    # Policy
    # policy_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.relu)  # FNN
    policy_hparam = dict(hidden_size=8, num_recurrent_layers=2)  # LSTM & GRU
    # policy = FNNPolicy(spec=env.spec, **policy_hparam)
    policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    # value_fcn_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.tanh)  # FNN
    value_fcn_hparam = dict(hidden_size=8, num_recurrent_layers=1)  # LSTM & GRU
    # value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.998,
        lamda=0.95,
        num_epoch=10,
        batch_size=200,
        lr=1e-3,
        # max_grad_norm=5.,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=30*env.max_steps,
        num_sampler_envs=20,
        num_epoch=10,
        value_fcn_coeff=0.7,
        entropy_coeff=5e-5,
        eps_clip=0.15,
        batch_size=200,
        std_init=env.domain_param['tau_max']*0.6,
        lr=5e-4,
        # max_grad_norm=5.,
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
    algo.train(seed=ex_dir.seed)
