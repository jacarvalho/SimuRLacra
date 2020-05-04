"""
Train an agent to solve the Quanser Qube environment using Proximal Policy Optimization.
"""
import torch as to

from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.environments.rcspysim.planar_3_link import Planar3LinkTASim
from pyrado.spaces import ValueFunctionSpace
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.rnn import LSTMPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(Planar3LinkTASim.name, PPO.name, 'lstm_actnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        dt=1/50.,
        max_steps=1000,
        max_dist_force=None,
        position_mps=True,
        taskCombinationMethod='sum',
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeVelocities=True,
        observeForceTorque=True,
        observeCollisionCost=False,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeGoalDistance=True,
        observeDynamicalSystemDiscrepancy=False,
    )
    env = Planar3LinkTASim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)
    policy = LSTMPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_size=16, num_recurrent_layers=1)
    value_fcn = LSTMPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.9995,
        lamda=0.960,
        num_epoch=10,
        batch_size=150,
        lr=7.e-4,
        standardize_adv=False,
        # max_grad_norm=5,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
        min_steps=20*env.max_steps,
        num_epoch=3,
        eps_clip=0.125,
        batch_size=150,
        std_init=0.7,
        lr=1e-4,
        num_sampler_envs=8,
        # max_grad_norm=5,
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed, snapshot_mode='best')
