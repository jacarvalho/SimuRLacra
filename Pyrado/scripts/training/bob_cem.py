"""
Train an agent to solve the Ball-on-Beam environment using Cross-Entropy Method .
"""
from pyrado.algorithms.cem import CEM
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, RBFFeat, identity_feat, sin_feat
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, CEM.name, 'rbf', seed=101)

    # Environment
    env_hparams = dict(dt=1/50., max_steps=300)
    env = BallOnBeamSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([identity_feat, sin_feat])
        # feats=FeatureStack([RBFFeat(num_feat_per_dim=7, bounds=env.obs_space.bounds, scale=None)])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=100,
        num_rollouts=12,
        num_is_samples=20,
        expl_std_init=0.5,
        expl_std_min=0.02,
        extra_expl_std_init=1.,
        extra_expl_decay_iter=5,
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
