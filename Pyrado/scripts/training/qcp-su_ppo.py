"""
Train an agent to solve the Quanser Qube environment using Proximal Policy Optimization.
"""
import torch as to
from torch.optim import lr_scheduler as scheduler

from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNN, FNNPolicy
from pyrado.policies.rnn import LSTMPolicy, GRUPolicy
from pyrado.utils.data_types import EnvSpec

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    # ex_dir = setup_experiment(QCartPoleSwingUpSim.name, PPO.name, 'fnn_actnorm', seed=1001)
    # ex_dir = setup_experiment(QCartPoleSwingUpSim.name, PPO.name, 'lstm_actnorm', seed=1001)
    ex_dir = setup_experiment(QCartPoleSwingUpSim.name, PPO.name, 'gru_actnorm', seed=1001)

    # Environment
    env_hparams = dict(dt=1/500., max_steps=4000, long=False)
    env = QCartPoleSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    # policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu)  # FNN
    policy_hparam = dict(hidden_size=64, num_recurrent_layers=1)  # LSTM & GRU
    # policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    # value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    value_fcn_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    # value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = LSTMPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.9995,
        lamda=0.98,
        num_epoch=5,
        batch_size=400,
        max_grad_norm=5,
        lr=5e-4,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
        min_steps=20*env.max_steps,
        num_epoch=5,
        eps_clip=0.1,
        batch_size=200,
        std_init=0.8,
        lr=2e-4,
        num_sampler_envs=16,
        max_grad_norm=5.,
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

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
