"""
Train an agent to solve the Ball-on-Plate environment using Cross-Entropy Method.
"""
from pyrado.algorithms.cem import CEM
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate5DSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnPlate5DSim.name, CEM.name, '', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',
        dt=1/100.,
        max_steps=500
    )
    env = BallOnPlate5DSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([identity_feat])
        # feats=FeatureStack([RBFFeat(num_feat_per_dim=7, bounds=env.obs_space.bounds, scale=None)])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=50,
        num_rollouts=36,
        num_is_samples=10,
        expl_std_init=2.,
        expl_std_min=0.02,
        full_cov=False,
        symm_sampling=False,
        num_sampler_envs=10,
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
