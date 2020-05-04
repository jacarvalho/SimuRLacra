"""
Train an agent to solve the Planar-3-Link task using Activation Dynamics Networks and Hill Climbing.
"""
import torch as to

from pyrado.algorithms.hc import HCNormal
from pyrado.environments.rcspysim.planar_box_fiddle import PlanarBoxFiddleIKSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.utils.data_types import EnvSpec
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PlanarBoxFiddleIKSim.name, 'fnn', '', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxFiddle.xml',
        dt=1/100.,
        max_steps=600,
        max_dist_force=None,
        taskCombinationMethod='mean',
    )
    env = PlanarBoxFiddleIKSim(**env_hparams)

    # Policy
    policy_hparam = dict(hidden_sizes=[16], hidden_nonlin=to.tanh)
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Algorithm (simple hill climbing)
    # algo_hparam = dict(
    #     max_iter=50,
    #     pop_size=250,
    #     expl_factor=1.1,
    #     num_rollouts=1,
    #     expl_std_init=1.0,
    #     num_sampler_envs=8,
    # )
    # algo = HCNormal(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    # save_list_of_dicts_to_yaml([
    #     dict(env=env_hparams, seed=ex_dir.seed),
    #     dict(FNN=policy_hparam),
    #     dict(algo=algo_hparam, algo_name=algo.name)],
    #     ex_dir
    # )
    # end hill climbing
    
    # Critic
    value_fcn_hparam = dict(hidden_sizes=[16], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=5,
        batch_size=100,
        lr=1e-3,
        standardize_adv=False,
        # max_grad_norm=5,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=10*env.max_steps,  # good: 30 times rollout length
        num_epoch=5,
        eps_clip=0.1,
        batch_size=100,
        std_init=1.0,
        lr=5e-4,
        num_sampler_envs=40,
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
    algo.train(seed=ex_dir.seed)
