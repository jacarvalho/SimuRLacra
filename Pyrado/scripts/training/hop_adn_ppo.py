"""
Train an agent to solve the Hopper environment using Proximal Policy Optimization.
"""
import torch as to

from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.policies.adn import ADNPolicy, pd_capacity_21
from pyrado.spaces import ValueFunctionSpace
from pyrado.environments.mujoco.openai_hopper import HopperSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(HopperSim.name, PPO.name, 'adn_lin', seed=1001)

    # Environment
    env_hparams = dict()
    env = HopperSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        # obs_layer=FNN(input_size=env.obs_space.flat_dim,
        #               output_size=env.act_space.flat_dim,
        #               hidden_sizes=[32, 32],
        #               hidden_nonlin=to.tanh),
        tau_init=1.,
        tau_learnable=True,
        capacity_learnable=False,
        activation_nonlin=to.tanh,
        potentials_dyn_fcn=pd_capacity_21,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

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
        lr=3e-4,
        num_sampler_envs=12,
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
    algo.train(seed=ex_dir.seed, snapshot_mode='best')
