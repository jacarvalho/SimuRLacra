"""
Train an agent to solve the Quanser Qube environment using Ensemble Policy Optimization.
"""
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.epopt import EPOpt
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.domain_randomization.default_randomizers import get_uniform_masses_lengths_randomizer_qq
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import LSTMPolicy, GRUPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    # ex_dir = setup_experiment(QQubeSim.name, 'epopt_ppo', 'gru_actnorm_dr-masses-lengths_hs-20th', seed=1)
    ex_dir = setup_experiment(QQubeSim.name, 'epopt_ppo', 'fnn_actnorm_dr-masses-lengths_hs-10th', seed=1)

    # Environment
    env_hparams = dict(dt=1/250., max_steps=1500)
    env = QQubeSim(**env_hparams)
    env = ActNormWrapper(env)
    # env = ActDelayWrapper(env)
    # randomizer = randomizer_uniform_masses_lengths()
    # randomizer.add_domain_params(UniformDomainParam(name='act_delay', mean=15, halfspan=15, clip_lo=0, roundint=True))
    env = DomainRandWrapperLive(env, get_uniform_masses_lengths_randomizer_qq(frac_halfspan=10.))

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.relu)  # FNN
    # policy_hparam = dict(hidden_size=64, num_recurrent_layers=1)  # LSTM & GRU
    policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    # policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.relu)  # FNN
    # value_fcn_hparam = dict(hidden_size=64, num_recurrent_layers=1)  # LSTM & GRU
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = LSTMPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=6,
        batch_size=150,
        lr=3.7e-4,
        max_grad_norm=5,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Subroutine
    subroutine_hparam = dict(
        max_iter=600,
        min_steps=30*env.max_steps,
        num_sampler_envs=6,
        num_epoch=2,
        eps_clip=0.05,
        batch_size=150,
        std_init=0.84,
        lr=5.7e-4,
        max_grad_norm=5.,
    )
    ppo = PPO(ex_dir, env, policy, critic, **subroutine_hparam)

    # Meta-Algorithm
    epopt_hparam = dict(skip_iter=300, epsilon=0.5, gamma=critic.gamma)
    algo = EPOpt(ppo, **epopt_hparam)

    # Save the hyper-parameters
    sm = 'best'
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed, snapshot_mode=sm),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(subroutine=subroutine_hparam, subroutine_name=PPO.name),
        dict(algo=epopt_hparam, algo_name=EPOpt.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(
        snapshot_mode=sm, seed=ex_dir.seed,
        # load_dir=osp.join(pyrado.TEMP_DIR, QQubeSim.name, 'epopt_ppo', '')
    )
