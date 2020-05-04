"""
Train an agent to solve the Ball-on-Beam environment using Relative Entropy Search.
"""
from pyrado.algorithms.reps import REPS
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, const_feat, identity_feat, squared_feat, sign_feat, qubic_feat, \
    RandFourierFeat
from pyrado.policies.linear import LinearPolicy

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PendulumSim.name, REPS.name, 'const-lin-sq-sign', seed=1001)
    # ex_dir = setup_experiment(PendulumSim.name, REPS.name, 'fourier', seed=1001)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=1500)
    env = PendulumSim(**env_hparams)
    # env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([const_feat, identity_feat, sign_feat, squared_feat, qubic_feat])
        # feats=FeatureStack([RandFourierFeat(env.obs_space.flat_dim, num_feat=200, bandwidth=env.obs_space.bound_up)])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        eps=0.15,
        gamma=0.995,
        pop_size=1000,
        num_rollouts=1,
        expl_std_init=2.0,
        expl_std_min=0.02,
        num_sampler_envs=20,
        num_epoch_dual=500,
        use_map=True,
        grad_free_optim=True,
        lr_dual=5e-4,
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
