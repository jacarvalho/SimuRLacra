"""
Learn the domain parameter distribution of masses and lengths of the Quanser Qube while using a handcrafted
randomization for the remaining domain parameters
"""
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.algorithms.ppo import PPO
from pyrado.domain_randomization.default_randomizers import get_zero_var_randomizer, get_default_domain_param_map_qq
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import LSTMPolicy, GRUPolicy
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.experiments import wrap_like_other_env


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, f'{BayRn.name}_{PPO.name}',
                              f'{FNNPolicy.name}_actnorm_dr-Mp-Mr-Lp-Lr', seed=111)

    # Environments
    env_hparams = dict(dt=1/100., max_steps=600)
    env_sim = QQubeSim(**env_hparams)
    env_sim = ActNormWrapper(env_sim)
    env_sim = DomainRandWrapperLive(env_sim, get_zero_var_randomizer(env_sim))
    dp_map = get_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    env_real = QQubeReal(**env_hparams)
    env_real = wrap_like_other_env(env_real, env_sim)

    # Policy
    policy_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    # policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    policy = FNNPolicy(spec=env_sim.spec, **policy_hparam)
    # policy = RNNPolicy(spec=env_sim.spec, **policy_hparam)
    # policy = LSTMPolicy(spec=env_sim.spec, **policy_hparam)
    # policy = GRUPolicy(spec=env_sim.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.tanh)  # FNN
    # value_fcn_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    value_fcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = GRUPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.9885,
        lamda=0.9648,
        num_epoch=2,
        batch_size=60,
        standardize_adv=False,
        lr=5.792e-4,
        max_grad_norm=1.,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Subroutine
    subroutine_hparam = dict(
        max_iter=300,
        min_steps=23*env_sim.max_steps,
        num_sampler_envs=16,
        num_epoch=7,
        eps_clip=0.0744,
        batch_size=60,
        std_init=0.9074,
        lr=3.446e-04,
        max_grad_norm=1.,
    )
    ppo = PPO(ex_dir, env_sim, policy, critic, **subroutine_hparam)

    # Set the boundaries for the GP
    dp_nom = QQubeSim.get_nominal_domain_param()
    # bounds = to.tensor(
    #     [[0.8*dp_nom['Mp'], dp_nom['Mp']/1000],
    #      [1.2*dp_nom['Mp'], dp_nom['Mp']/10]])
    bounds = to.tensor(
        [[0.8*dp_nom['Mp'], dp_nom['Mp']/1000, 0.8*dp_nom['Mr'], dp_nom['Mr']/1000,
          0.8*dp_nom['Lp'], dp_nom['Lp']/1000, 0.8*dp_nom['Lr'], dp_nom['Lr']/1000],
         [1.2*dp_nom['Mp'], dp_nom['Mp']/10, 1.2*dp_nom['Mr'], dp_nom['Mr']/10,
          1.2*dp_nom['Lp'], dp_nom['Lp']/10, 1.2*dp_nom['Lr'], dp_nom['Lr']/10]])

    # policy_init = to.load(osp.join(pyrado.EXP_DIR, QQubeSim.name, PPO.name, 'EXP_NAME', 'policy.pt'))
    # valuefcn_init = to.load(osp.join(pyrado.EXP_DIR, QQubeSim.name, PPO.name, 'EXP_NAME', 'valuefcn.pt'))

    # Algorithm
    bayrn_hparam = dict(
        thold_succ=300.,
        max_iter=15,
        acq_fc='EI',
        # acq_param=dict(beta=0.2),
        acq_restarts=500,
        acq_samples=1000,
        num_init_cand=10,
        warmstart=False,
        # policy_param_init=policy_init.param_values.data,
        # valuefcn_param_init=valuefcn_init.param_values.data,
    )

    # Save the environments and the hyper-parameters (do it before the init routine of BayRn)
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(subroutine=subroutine_hparam, subroutine_name=PPO.name),
        dict(algo=bayrn_hparam, algo_name=BayRn.name, dp_map=dp_map)],
        ex_dir
    )

    algo = BayRn(ex_dir, env_sim, env_real, subroutine=ppo, bounds=bounds, **bayrn_hparam)

    # Jeeeha
    algo.train(
        snapshot_mode='latest',
        seed=ex_dir.seed,
    )

    # Train the policy on the most lucrative domain
    BayRn.train_argmax_policy(
        ex_dir, env_sim, ppo, num_restarts=500, num_samples=1000,
        # policy_param_init=policy.param_values.data,
        # valuefcn_param_init=critic.value_fcn.param_values.data
    )
