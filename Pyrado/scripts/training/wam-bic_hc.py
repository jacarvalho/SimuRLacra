"""
Train an agent to solve the WAM Ball-in-cup environment using Hill Climbing.
"""
from pyrado.algorithms.hc import HCNormal
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import DualRBFLinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(WAMBallInCupSim.name, HCNormal.name, '', seed=101)

    # Environment
    env_hparams = dict(
        max_steps=1750,
        task_args=dict(final_factor=0.05)
    )
    env = WAMBallInCupSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        rbf_hparam=dict(num_feat_per_dim=7, bounds=(0., 1.), scale=None),
        dim_mask=2
    )
    policy = DualRBFLinearPolicy(env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=5*policy.num_param,
        expl_factor=1.05,
        num_rollouts=1,
        expl_std_init=0.5,
        num_sampler_envs=6,
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
