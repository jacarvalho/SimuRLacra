"""
Train an agent to solve the Ball-on-Beam environment using Relative Entropy Search.
"""
from pyrado.algorithms.power import PoWER
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat, RBFFeat
from pyrado.policies.linear import LinearPolicy

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, PoWER.name, 'lin-ident-sin', seed=101)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=500)
    env = BallOnBeamSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        # feats=FeatureStack([RBFFeat(num_feat_per_dim=50, bounds=env.obs_space.bounds, scale=None)])
        feats=FeatureStack([identity_feat, sin_feat])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=200,
        num_rollouts=10,
        num_is_samples=100,
        expl_std_init=1.5,
        expl_std_min=0.02,
        symm_sampling=False,
        num_sampler_envs=4,
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
