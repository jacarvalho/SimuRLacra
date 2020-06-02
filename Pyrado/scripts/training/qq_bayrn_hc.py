"""
Learn the domain parameter distribution of masses and lengths of the Quanser Qube while using a handcrafted
randomization for the remaining domain parameters
"""
import torch as to

from pyrado.algorithms.hc import HCNormal
from pyrado.domain_randomization.default_randomizers import get_zero_var_randomizer, get_default_domain_param_map_qq
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.utils.experiments import wrap_like_other_env


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, f'{BayRn.name}_{HCNormal.name}-sim2sim',
                              '100Hz_ectrl_dr-Mp+Mr+_wider', seed=111)

    # Environments
    env_hparams = dict(dt=1/100., max_steps=600)
    env_sim = QQubeSim(**env_hparams)
    env_sim = DomainRandWrapperLive(env_sim, get_zero_var_randomizer(env_sim))
    dp_map = get_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    env_real = QQubeSim(**env_hparams)
    env_real.domain_param = dict(Mp=0.026, Mr=0.097)
    # env_real.domain_param = dict(Mp=0.024*0.95, Mr=0.095*1.05)
    # env_real = QQubeReal(**env_hparams)
    env_real = wrap_like_other_env(env_real, env_sim)

    # Policy
    policy_hparam = dict(energy_gain=0.587, ref_energy=0.827, acc_max=10.)
    policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec, **policy_hparam)

    # Subroutine
    subroutine_hparam = dict(
        max_iter=25,
        pop_size=50,
        expl_factor=1.1,
        num_rollouts=20,
        expl_std_init=2.0,
        num_sampler_envs=32,
    )
    hc = HCNormal(ex_dir, env_sim, policy, **subroutine_hparam)

    # Set the boundaries for the GP
    dp_nom = QQubeSim.get_nominal_domain_param()
    # bounds = to.tensor(
    #     [[0.8*dp_nom['Mp'], dp_nom['Mp']/2000],
    #      [1.2*dp_nom['Mp'], dp_nom['Mp']/1000]])
    bounds = to.tensor(
        [[0.8*dp_nom['Mp'], dp_nom['Mp']/5000, 0.8*dp_nom['Mr'], dp_nom['Mr']/5000],
         [1.2*dp_nom['Mp'], dp_nom['Mp']/4999, 1.2*dp_nom['Mr'], dp_nom['Mr']/4999]])
    # bounds = to.tensor(
    #     [[0.9*dp_nom['Mp'], dp_nom['Mp']/1000, 0.9*dp_nom['Mr'], dp_nom['Mr']/1000,
    #       0.9*dp_nom['Lp'], dp_nom['Lp']/1000, 0.9*dp_nom['Lr'], dp_nom['Lr']/1000],
    #      [1.1*dp_nom['Mp'], dp_nom['Mp']/20, 1.1*dp_nom['Mr'], dp_nom['Mr']/20,
    #       1.1*dp_nom['Lp'], dp_nom['Lp']/20, 1.1*dp_nom['Lr'], dp_nom['Lr']/20]])

    # Algorithm
    bayrn_hparam = dict(
        max_iter=15,
        acq_fc='UCB',
        acq_param=dict(beta=0.2),
        acq_restarts=500,
        acq_samples=1000,
        num_init_cand=5,
        warmstart=False,
        num_eval_rollouts_real=1000 if isinstance(env_real, QQubeSim) else 5,
    )

    # Save the environments and the hyper-parameters (do it before the init routine of BayRn)
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(subroutine=subroutine_hparam, subroutine_name=HCNormal.name),
        dict(algo=bayrn_hparam, algo_name=BayRn.name, dp_map=dp_map)],
        ex_dir
    )

    algo = BayRn(ex_dir, env_sim, env_real, subroutine=hc, bounds=bounds, **bayrn_hparam)

    # Jeeeha
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
