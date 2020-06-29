"""
Train an agent to solve the Ball-on-Beam environment using Asynchronous Actor-Critic.
"""
import torch as to
from torch.optim import lr_scheduler as scheduler

from pyrado.utils.data_types import EnvSpec
from pyrado.algorithms.a2c import A2C
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat, RandFourierFeat
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.linear import LinearPolicy

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, A2C.name, LinearPolicy.name, seed=1001)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=500)
    env = BallOnBeamSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        # feats=FeatureStack([RandFourierFeat(env.obs_space.flat_dim, num_feat=100, bandwidth=env.obs_space.bound_up)])
        feats=FeatureStack([identity_feat, sin_feat])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.99,
        lamda=0.95,
        batch_size=100,
        standardize_adv=False,
        lr_scheduler=scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.99)
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=10000,
        num_sampler_envs=4,
        value_fcn_coeff=0.7,
        entropy_coeff=4e-5,
        batch_size=100,
        std_init=0.8,
        lr=2e-3,
        lr_scheduler=scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.99)
    )
    algo = A2C(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
