"""
Train an agent to solve the Quanser Qube task using Natural Evolution Strategies.
"""
from pyrado.algorithms.nes import NES
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.policies.features import *
from pyrado.policies.fnn import FNNPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, NES.name, 'fnn_actnorm', seed=1001)

    # Environments
    env_hparams = dict(dt=1/250., max_steps=1500)
    env = QQubeSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
        pop_size=1000,
        num_rollouts=6,
        eta_mean=2.,
        eta_std=None,
        expl_std_init=0.2,
        symm_sampling=False,
        transform_returns=True,
        base_seed=ex_dir.seed,
        num_sampler_envs=12,
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
