"""
Solve the pendulum environment with Bayesian Domain Adaptation (sim-to-sim)
"""
import numpy as np
import torch as to

from pyrado.algorithms.hc import HCNormal
from pyrado.domain_randomization.default_randomizers import get_zero_var_randomizer, get_default_domain_param_map_pend
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, abs_feat, sign_feat, squared_feat, qubic_feat
from pyrado.policies.linear import LinearPolicy
from pyrado.utils.experiments import wrap_like_other_env


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PendulumSim.name, BayRn.name, 'lin-abs-sign-sq-qub_actnorm_one', seed=1001)
    # ex_dir = setup_experiment(PendulumSim.name, BayRn.name, 'const-lin_actnorm_all', seed=1)

    # Environments
    env_hparams = dict(dt=1/100., max_steps=1500)
    env_sim = PendulumSim(**env_hparams)
    env_sim = ActNormWrapper(env_sim)
    env_sim = DomainRandWrapperLive(env_sim, get_zero_var_randomizer(env_sim))
    dp_map = get_default_domain_param_map_pend()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    env_real = PendulumSim(**env_hparams)
    env_real = wrap_like_other_env(env_real, env_sim)
    # Make the target domain different
    env_real.domain_param = dict(tau_max=6.5)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([identity_feat, abs_feat, sign_feat, squared_feat, qubic_feat])
    )
    policy = LinearPolicy(spec=env_sim.spec, **policy_hparam)

    # Subroutine
    algo_hparam = dict(
        max_iter=20,
        pop_size=500,
        expl_factor=1.1,
        num_rollouts=20,
        expl_std_init=1.,
        num_sampler_envs=20,
    )
    hc = HCNormal(ex_dir, env_sim, policy, **algo_hparam)

    # Set the boundaries for the GP
    dp_nom = PendulumSim.get_nominal_domain_param()
    bounds = to.tensor(
        [[0.5*dp_nom['tau_max'], dp_nom['tau_max']/1000],
         [2.0*dp_nom['tau_max'], dp_nom['tau_max']/100]])

    # Algorithm
    bayrn_hparam = dict(
        max_iter=10,
        acq_fc='UCB',
        acq_restarts=200,
        acq_samples=1000,
        acq_param=dict(beta=0.2),
        num_init_cand=5,
        num_eval_rollouts=100,
        warmstart=True,
    )

    # Save the environments and the hyper-parameters (do it before the init routine of BayRn)
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        dict(algo=bayrn_hparam, algo_name=algo.name, dp_map=dp_map)],
        # dict(algo=bayrn_hparam, algo_name=algo.name, randomizer_fcn=randomizer_all)],
        ex_dir
    )

    algo = BayRn(ex_dir, env_sim, env_real, hc, bounds, **bayrn_hparam)

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
