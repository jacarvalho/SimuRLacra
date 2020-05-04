"""
Learn the domain parameter distribution of masses and lengths of the Quanser Qube while using a handcrafted
randomization for the remaining domain parameters
"""
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.nes import NES
from pyrado.domain_randomization.default_randomizers import get_default_domain_param_map_qq, get_zero_var_randomizer
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, sign_feat, abs_feat, squared_feat, qubic_feat, \
    bell_feat
from pyrado.policies.linear import LinearPolicy
from pyrado.utils.experiments import wrap_like_other_env

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, 'bayrn_nes', 'fnn_actnorm_dr-Mp-Mr-Lp-Lr', seed=111)

    # Environments
    env_hparams = dict(dt=1/250., max_steps=1500)
    env_sim = QQubeSim(**env_hparams)
    env_sim = ActNormWrapper(env_sim)
    env_sim = DomainRandWrapperLive(env_sim, get_zero_var_randomizer(env_sim))
    dp_map = get_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    env_real = QQubeReal(**env_hparams)
    env_real = wrap_like_other_env(env_real, env_sim)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([identity_feat, sign_feat, abs_feat, squared_feat, qubic_feat, bell_feat])
    )
    policy = LinearPolicy(spec=env_sim.spec, **policy_hparam)

    # Subroutine
    subroutine_hparam = dict(
        max_iter=3,
        pop_size=3,
        num_rollouts=8,
        eta_mean=2.,
        eta_std=None,
        expl_std_init=0.2,
        symm_sampling=False,
        transform_returns=True,
        base_seed=ex_dir.seed,
        num_sampler_envs=1,
    )
    nes = NES(ex_dir, env_sim, policy, **subroutine_hparam)

    # Set the boundaries for the GP
    dp_nom = QQubeSim.get_nominal_domain_param()
    bounds = to.tensor(
        [[0.9*dp_nom['Mp'], dp_nom['Mp']/1000, 0.9*dp_nom['Mr'], dp_nom['Mr']/1000,
          0.9*dp_nom['Lp'], dp_nom['Lp']/1000, 0.9*dp_nom['Lr'], dp_nom['Lr']/1000],
         [1.1*dp_nom['Mp'], dp_nom['Mp']/20, 1.1*dp_nom['Mr'], dp_nom['Mr']/20,
          1.1*dp_nom['Lp'], dp_nom['Lp']/20, 1.1*dp_nom['Lr'], dp_nom['Lr']/20]])

    policy_init = to.load(osp.join(pyrado.PERMA_DIR, QQubeSim.name, NES.name,
                                   '2020-01-06_16-53-50--fnn_actnorm--perfect', 'policy.pt'))

    # Algorithm
    bayrn_hparam = dict(
        max_iter=10,
        acq_fc='UCB',
        acq_param=dict(beta=0.1),
        acq_restarts=500,
        acq_samples=1000,
        num_init_cand=10,
        warmstart=True,
        num_eval_rollouts=100 if isinstance(env_real, QQubeSim) else 5,
        policy_param_init=policy_init.param_values.data
    )

    # Save the environments and the hyper-parameters (do it before the init routine of BayRn)
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(subroutine_algo=subroutine_hparam, subroutine_name=NES.name),
        dict(algo=bayrn_hparam, algo_name=BayRn.name, dp_map=dp_map)],
        ex_dir
    )

    algo = BayRn(ex_dir, env_sim, env_real, subroutine=nes, bounds=bounds, **bayrn_hparam)

    # Jeeeha
    algo.train(
        snapshot_mode='best',
        seed=ex_dir.seed,
        # load_dir=osp.join(pyrado.EXP_DIR, QQubeSim.name, 'bayrn_nes', '')
    )
