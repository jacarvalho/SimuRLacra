"""
Train an agent to solve the Quanser Qube environment using Active Domain Randomization.
"""

import torch as to

from pyrado.algorithms.adr import ADR
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, f'{ADR.name}-{PPO.name}', 'fnn_actnorm', seed=1001)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=600)
    env = QQubeSim(**env_hparams)
    env = ActNormWrapper(env)
    env = ActDelayWrapper(env)

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.99,
        lamda=0.95,
        num_epoch=1,
        batch_size=64,
        lr=5e-4,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Subroutine
    algo_hparam = dict(
        max_iter=50,
        min_steps=1000,
        num_sampler_envs=1,
        num_epoch=1,
        eps_clip=0.1,
        batch_size=64,
        std_init=2.0,
        lr=8e-4
    )
    ppo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Algorithm
    svpg_actor_hparam = dict(
        hidden_sizes=[64, 64],
        hidden_nonlin=to.relu,
    )
    svpg_value_fcn_hparam = dict(
        hidden_sizes=[64, 64],
        hidden_nonlin=to.tanh,
    )
    svpg_critic_hparam = dict(
        gamma=0.995,
        lamda=1.,
        num_epoch=1,
        lr=1e-4,
        standardize_adv=False,
    )
    svpg_particle_hparam = dict(actor=svpg_actor_hparam, value_fcn=svpg_value_fcn_hparam, critic=svpg_critic_hparam)
    adr_hparam = dict(
        max_iter=200,
        num_svpg_particles=10,
        num_discriminator_epoch=10,
        batch_size=128,
        randomized_params=[],
        num_sampler_envs=4,
    )
    adr = ADR(ex_dir, env, subroutine=ppo, svpg_particle_hparam=svpg_particle_hparam, **adr_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(subroutine=algo_hparam, subroutine_name=PPO.name),
        dict(algo=adr_hparam, algo_name=ADR.name, svpg_particle_hparam=svpg_particle_hparam)],
        ex_dir
    )

    adr.train(snapshot_mode='best', seed=ex_dir.seed)
