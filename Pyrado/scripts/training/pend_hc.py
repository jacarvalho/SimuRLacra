"""
Train an agent to solve the inverted pendulum  environment using Hill Climbing.
"""
from pyrado.algorithms.hc import HCNormal
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, abs_feat, sign_feat, squared_feat, qubic_feat,\
    RandFourierFeat
from pyrado.policies.linear import LinearPolicy

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PendulumSim.name, HCNormal.name, 'ident-abs-sign-sq-qub', seed=1001)
    # ex_dir = setup_experiment(PendulumSim.name, HCNormal.name, 'fourier', seed=1001)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=1500)
    env = PendulumSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([identity_feat, abs_feat, sign_feat, squared_feat, qubic_feat])
        # feats=FeatureStack([RandFourierFeat(env.obs_space.flat_dim, num_feat=100, bandwidth=env.obs_space.bound_up)])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=500,
        expl_factor=1.1,
        num_rollouts=1,
        expl_std_init=1.,
        num_sampler_envs=20,
    )
    algo = HCNormal(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
