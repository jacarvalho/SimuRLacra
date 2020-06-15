"""
Train an agent to solve the Quanser Cart-Pole stabilization task using Cross-Entropy Method.
"""
from pyrado.algorithms.cem import CEM
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.policies.features import *
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QCartPoleStabSim.name, CEM.name, 'lin-ident', seed=1001)

    # Environments
    env_hparams = dict(dt=1/100., max_steps=500, long=False)
    env = QCartPoleStabSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(feats=FeatureStack([identity_feat]))
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=20,
        pop_size=50,
        num_rollouts=8,
        num_is_samples=10,
        expl_std_init=2.,
        expl_std_min=0.02,
        full_cov=True,
        symm_sampling=False,
        num_sampler_envs=8,
    )
    algo = CEM(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
