"""
Train an agent to solve the Quanser Qube environment using Proximal Policy Optimization.
"""
import torch as to

from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, PPO.name, 'fnn_actnorm')

    # Environment
    env_hparams = dict(dt=1/100., max_steps=600)
    env = QQubeSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)  # FNN
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)  # FNN
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.99,
        lamda=0.95,
        num_epoch=10,
        batch_size=64,
        lr=5e-4,
        max_grad_norm=1.,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=150,
        min_steps=30*env.max_steps,
        num_epoch=10,
        eps_clip=0.1,
        batch_size=64,
        std_init=1.0,
        lr=5e-4,
        num_sampler_envs=8,
        max_grad_norm=1.,
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode='best')
