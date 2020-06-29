"""
Train an agent to solve the One-Mass-Oscillator environment using Parameter-Exploring Policy Gradients.
"""
import numpy as np

from pyrado.algorithms.pepg import PEPG
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, const_feat, identity_feat
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(OneMassOscillatorSim.name, PEPG.name, LinearPolicy.name, seed=1001)

    # Environment
    env_hparams = dict(dt=1/50., max_steps=200)
    env = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([const_feat, identity_feat])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        num_rollouts=8,
        pop_size=60,
        expl_std_init=1.0,
        clip_ratio_std=0.05,
        normalize_update=False,
        transform_returns=True,
        lr=1e-2,
        num_sampler_envs=8,
    )
    algo = PEPG(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
