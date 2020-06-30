"""
Train an agent to solve the Box Shelving task task using Neural Fields and Proximal Policy Optimization.
"""
import numpy as np
import torch as to

from pyrado.algorithms.advantage import GAE
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
    ex_dir = setup_experiment(OneMassOscillatorSim.name, PPO.name, NFPolicy.name, seed=1001)

    # Environment
    env_hparams = dict(dt=1/50., max_steps=200)
    env = OneMassOscillatorSim(**env_hparams, task_args=dict(state_des=np.array([0.5, 0])))
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        hidden_size=5,
        conv_out_channels=1,
        mirrored_conv_weights=True,
        conv_kernel_size=3,
        conv_padding_mode='circular',
        init_param_kwargs=dict(bell=True),
        activation_nonlin=to.sigmoid,
        tau_init=1e-1,
        tau_learnable=False,
        kappa_init=1e-3,
        kappa_learnable=True,
        potential_init_learnable=True,
    )
    policy = NFPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=10,
        batch_size=256,
        standardize_adv=False,
        standardizer=None,
        # max_grad_norm=5.,
        lr=1e-3,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=10*env.max_steps,
        num_epoch=5,
        eps_clip=0.15,
        batch_size=256,
        std_init=0.6,
        # max_grad_norm=5.,
        lr=5e-4,
        num_sampler_envs=6,
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
