"""
Train an agent to solve the Box Shelving task task using Activation Dynamics Networks and Cross-Entropy Method.
"""
import numpy as np
import torch as to

from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.cem import CEM
from pyrado.algorithms.ppo import PPO
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.neural_fields import NFPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(OneMassOscillatorSim.name, f'nf-{CEM.name}', 'const-lin', seed=1001)

    # Environment
    env_hparams = dict(dt=1/50., max_steps=200)
    env = OneMassOscillatorSim(**env_hparams, task_args=dict(state_des=np.array([0.5, 0])))
    # env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        hidden_size=5,
        conv_out_channels=1,
        conv_kernel_size=5,
        conv_padding_mode='circular',
        activation_nonlin=to.tanh,
        tau_init=1e-1,
        tau_learnable=True,
    )
    policy = NFPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=10,
        batch_size=512,
        standardize_adv=False,
        standardizer=None,
        max_grad_norm=1.,
        lr=5e-4,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=20*env.max_steps,
        num_epoch=10,
        eps_clip=0.15,
        batch_size=512,
        max_grad_norm=1.,
        lr=5e-4,
        num_sampler_envs=8,
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
