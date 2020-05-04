"""
Train an agent to solve the Quanser Cart-Pole stabilization task using Natural Evolution Strategies.
"""
from pyrado.algorithms.nes import NES
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.policies.features import *
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QCartPoleStabSim.name, NES.name, 'lin-ident-sin-cos_actnorm', seed=1001)

    # Environments
    env_hparams = dict(dt=1/100., max_steps=500, long=False)
    env = QCartPoleStabSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(feats=FeatureStack([identity_feat, squared_feat]))
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=200,
        pop_size=None,
        num_rollouts=12,
        eta_mean=1.,
        eta_std=None,
        expl_std_init=0.5,
        symm_sampling=False,
        transform_returns=True,
        num_sampler_envs=16,
    )
    algo = NES(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
