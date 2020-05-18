"""
Train an agent to solve the WAM Ball-in-cup environment using Policy learning by Weighting Exploration with the Returns.
"""
import numpy as np

from pyrado.algorithms.power import PoWER
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import DualRBFLinearPolicy

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(WAMBallInCupSim.name, PoWER.name, seed=101)

    # Environment
    env_hparams = dict(
        max_steps=1750,
        task_args=dict(factor=1.)
    )
    env = WAMBallInCupSim(**env_hparams)

    # Policy
    rbf_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.]), np.array([1.])), scale=None)
    policy = DualRBFLinearPolicy(env.spec, rbf_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=200,
        num_rollouts=1,
        num_is_samples=20,
        expl_std_init=0.5,
        expl_std_min=0.05,
        num_sampler_envs=4,
    )
    algo = PoWER(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
