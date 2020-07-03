"""
Train an agent to solve the Quanser Cart-Pole stabilization task using Cross-Entropy Method.
"""
from pyrado.algorithms.cem import CEM
from pyrado.algorithms.power import PoWER
from pyrado.environments.pysim.quanser_qube import QQubeStabSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.policies.features import *
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeStabSim.name, CEM.name, 'ectrl', seed=1)

    # Environments
    env_hparams = dict(dt=1/500., max_steps=5000)
    env = QQubeStabSim(**env_hparams)
    # env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(feats=FeatureStack([identity_feat]))
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=50,
        num_rollouts=10,
        num_is_samples=10,
        expl_std_init=2.,
        expl_std_min=0.02,
        full_cov=True,
        symm_sampling=False,
        num_sampler_envs=12,
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

    print(policy.param_values)
