"""
Train an agent to solve the Quanser Ball-Balancer environment using
Simulation-based Policy Optimization with Transferability Assessment.
"""
from copy import deepcopy
from numpy import pi
import torch as to

from pyrado.algorithms.hc import HCNormal
from pyrado.algorithms.spota import SPOTA
from pyrado.domain_randomization.default_randomizers import get_default_randomizer
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat
from pyrado.policies.linear import LinearPolicy
from pyrado.sampling.sequences import *

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QBallBalancerSim.name, f'{SPOTA.name}-{HCNormal.name}',
                              f'{LinearPolicy.name}_obsnoise-s_actedlay-10', seed=1001)

    # Environment and domain randomization
    env_hparams = dict(dt=1/100., max_steps=500)
    env = QBallBalancerSim(**env_hparams)
    env = GaussianObsNoiseWrapper(env, noise_std=[1/180*pi, 1/180*pi, 0.005, 0.005,  # [rad, rad, m, m, ...
                                                  10/180*pi, 10/180*pi, 0.05, 0.05])  # ... rad/s, rad/s, m/s, m/s]
    # env = ObsPartialWrapper(env, mask=[0, 0, 0, 0, 1, 1, 0, 0])
    env = ActDelayWrapper(env)
    randomizer = get_default_randomizer(env)
    randomizer.add_domain_params(UniformDomainParam(name='act_delay', mean=5, halfspan=5, clip_lo=0, roundint=True))
    env = DomainRandWrapperBuffer(env, randomizer)

    # Policy
    policy_hparam = dict(feats=FeatureStack([identity_feat]))
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Initialize with Quanser's PD gains
    init_policy_param_values = to.tensor([[-14., 0, -14*3.45, 0, 0, 0, -14*2.11, 0],
                                          [0, -14., 0, -14*3.45, 0, 0, 0, -14*2.11]])

    # Algorithm
    subrtn_hparam_cand = dict(
        max_iter=100,
        num_rollouts=0,  # will be overwritten by SPOTA
        pop_size=50,
        expl_factor=1.1,
        expl_std_init=0.5,
    )
    subrtn_hparam_cand = subrtn_hparam_cand

    sr_cand = HCNormal(ex_dir, env, policy, **subrtn_hparam_cand)
    sr_refs = HCNormal(ex_dir, env, deepcopy(policy), **subrtn_hparam_cand)

    spota_hparam = dict(
        max_iter=10,
        alpha=0.05,
        beta=0.1,
        nG=20,
        nJ=120,
        ntau=5,
        nc_init=5,
        nr_init=1,
        sequence_cand=sequence_add_init,
        sequence_refs=sequence_const,
        warmstart_cand=True,
        warmstart_refs=True,
        cand_policy_param_init=init_policy_param_values,
        num_bs_reps=1000,
        studentized_ci=False,
    )
    algo = SPOTA(ex_dir, env, sr_cand, sr_refs, **spota_hparam)

    # Save the environments and the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(subroutine_cand=subrtn_hparam_cand, subroutine_refs=subrtn_hparam_cand, subroutine_name=HCNormal.name),
        dict(algo=spota_hparam, algo_name=SPOTA.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
