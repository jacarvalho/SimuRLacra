"""
Train an agent to solve the Ball-on-Beam environment using Relative Entropy Search.

.. note::
    The hyper-parameters are not tuned at all!
"""
from pyrado.algorithms.reps import REPS
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, RandFourierFeat, RBFFeat, identity_feat, sin_feat
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, REPS.name, 'lin-ident-sin', seed=1001)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=500)
    env = BallOnBeamSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        # feats=FeatureStack([RandFourierFeat(env.obs_space.flat_dim, num_feat=100, bandwidth=env.obs_space.bound_up)])
        # feats=FeatureStack([RBFFeat(num_feat_per_dim=20, bounds=env.obs_space.bounds, scale=0.8)]),
        feats=FeatureStack([identity_feat, sin_feat])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=200,
        eps=0.1,
        gamma=0.995,
        pop_size=20*policy.num_param,
        num_rollouts=10,
        expl_std_init=1.0,
        expl_std_min=0.02,
        num_epoch_dual=500,
        grad_free_optim=True,
        lr_dual=1e-4,
        use_map=False,
        num_sampler_envs=4,
    )
    algo = REPS(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
