"""
Train an agent to solve the WAM Ball-in-cup environment using Hill Climbing.
"""
import numpy as np

from pyrado.algorithms.hc import HCNormal
from pyrado.domain_randomization.domain_parameter import UniformDomainParam, NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import DualRBFLinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(WAMBallInCupSim.name, HCNormal.name,
                              f'{DualRBFLinearPolicy.name}_nom', seed=1001)
                              # f'{DualRBFLinearPolicy.name}_dr-cs-rl-m-jd-js', seed=1001)

    # Environment
    env_hparams = dict(
        max_steps=2000,
        task_args=dict(final_factor=0.01),
        fixed_initial_state=True,
    )
    env = WAMBallInCupSim(**env_hparams)

    # Randomizer
    # randomizer = DomainRandomizer(
    #     UniformDomainParam(name='cup_scale', mean=0.95, halfspan=0.05),
    #     NormalDomainParam(name='rope_length', mean=0.3, std=0.005),
    #     NormalDomainParam(name='ball_mass', mean=0.021, std=0.001),
    #     UniformDomainParam(name='joint_damping', mean=0.05, halfspan=0.05),
    #     UniformDomainParam(name='joint_stiction', mean=0.1, halfspan=0.1),
    # )
    # env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(
        rbf_hparam=dict(num_feat_per_dim=12, bounds=(0., 1.), scale=None),
        dim_mask=2
    )
    policy = DualRBFLinearPolicy(env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=10*policy.num_param,
        expl_factor=1.05,
        num_rollouts=1,
        expl_std_init=np.pi/12,
        num_sampler_envs=8,
    )
    algo = HCNormal(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
