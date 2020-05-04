"""
Learn the domain parameter distribution of masses and lengths of the Quanser Qube while using a handcrafted
randomization for the remaining domain parameters
"""
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
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.experiments import wrap_like_other_env


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, 'bayrn_ppo', 'fnn_actnorm_dr-masses-lengths', seed=401)

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
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    policy = FNNPolicy(spec=env_sim.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.99,
        lamda=0.95,
        num_epoch=10,
        batch_size=64,
        lr=5e-4,
        max_grad_norm=1.
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Subroutine
    subroutine_hparam = dict(
        max_iter=150,
        min_steps=30*env_sim.max_steps,
        num_sampler_envs=8,
        num_epoch=10,
        eps_clip=0.1,
        batch_size=64,
        lr=5e-4,
        max_grad_norm=1.
    )
    ppo = PPO(ex_dir, env_sim, policy, critic, **subroutine_hparam)

    # Set the boundaries for the GP
    dp_nom = QQubeSim.get_nominal_domain_param()
    bounds = to.tensor(
        [[0.8*dp_nom['Mp'], dp_nom['Mp']/500, 0.8*dp_nom['Mr'], dp_nom['Mr']/500,
          0.8*dp_nom['Lp'], dp_nom['Lp']/500, 0.8*dp_nom['Lr'], dp_nom['Lr']/500],
         [1.2*dp_nom['Mp'], dp_nom['Mp']/5, 1.2*dp_nom['Mr'], dp_nom['Mr']/5,
          1.2*dp_nom['Lp'], dp_nom['Lp']/5, 1.2*dp_nom['Lr'], dp_nom['Lr']/5]])

    # Algorithm
    bayrn_hparam = dict(
        max_iter=15,
        acq_fc='EI',
        acq_restarts=500,
        acq_samples=1000,
        num_init_cand=10,
        num_eval_rollouts=3,
        warmstart=False,
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
    )
