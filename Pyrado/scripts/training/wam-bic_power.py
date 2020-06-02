"""
Train an agent to solve the WAM Ball-in-cup environment using Policy learning by Weighting Exploration with the Returns.
"""
import numpy as np

from pyrado.algorithms.power import PoWER
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.domain_randomization.default_randomizers import get_default_randomizer_wambic
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import DualRBFLinearPolicy

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(WAMBallInCupSim.name, PoWER.name, 'randomized', seed=101)

    # Environment
    env_hparams = dict(
        max_steps=1000,
        task_args=dict(factor=0.2)
    )
    env = WAMBallInCupSim(**env_hparams)

    # Simple Randomizer
    # randomizer = DomainRandomizer(
    #     NormalDomainParam(name='cup_scale', mean=1.5, std=0.3, clip_lo=1., clip_up=2.5)
    # )
    env = DomainRandWrapperLive(env, get_default_randomizer_wambic())

    # Policy
    policy_hparam = dict(
        rbf_hparam=dict(num_feat_per_dim=7, bounds=(0., 1.), scale=None),
        dim_mask=2
    )
    policy = DualRBFLinearPolicy(env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=15,
        pop_size=35,
        num_rollouts=5,
        num_is_samples=15,
        expl_std_init=1.0,
        expl_std_min=0.05,
        num_sampler_envs=8,
    )
    algo = PoWER(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
