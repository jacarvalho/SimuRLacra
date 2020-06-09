"""
Train an agent to solve the WAM Ball-in-cup environment using Hill Climbing.
"""
from pyrado.algorithms.hc import HCNormal
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import DualRBFLinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(WAMBallInCupSim.name, HCNormal.name, 'randomized', seed=101)

    # Environment
    env_hparams = dict(
        max_steps=1500,
        task_args=dict(factor=0.05)
    )
    env = WAMBallInCupSim(**env_hparams)

    # Simple Randomizer
    dp_nom = WAMBallInCupSim.get_nominal_domain_param()
    randomizer = DomainRandomizer(
        NormalDomainParam(name='cup_scale', mean=1.5, std=0.2, clip_lo=1., clip_up=2.)
    )
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(
        rbf_hparam=dict(num_feat_per_dim=7, bounds=(0., 1.), scale=None),
        dim_mask=2
    )
    policy = DualRBFLinearPolicy(env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=4*policy.num_param,
        expl_factor=1.05,
        num_rollouts=3,
        expl_std_init=0.1,
        num_sampler_envs=4,
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
