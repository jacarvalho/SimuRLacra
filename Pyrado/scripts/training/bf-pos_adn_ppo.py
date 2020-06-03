"""
Train an agent to solve the Hopper environment using Proximal Policy Optimization.
"""
import torch as to

from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.box_flipping import BoxFlippingPosMPsSim
from pyrado.policies.adn import ADNPolicy, pd_capacity_21
from pyrado.spaces import ValueFunctionSpace
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxFlippingPosMPsSim.name, f'adn-{PPO.name}', 'trqctrl_obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxFlipping_trqCtrl.xml',  # gBoxFlipping_posCtrl.xml or gBoxFlipping_trqCtrl.xml
        dt=1/100.,
        max_steps=1500,
        ref_frame='world',  # world, table, or box
        collisionConfig={'file': 'collisionModel.xml'},
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        checkJointLimits=True,
        collisionAvoidanceIK=False,
        observeManipulators=False,
        observeBoxOrientation=False,
        observeVelocities=False,
        observeForceTorque=True,
        observeCollisionCost=False,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=False,
        observeDSGoalDistance=False,
    )
    env = BoxFlippingPosMPsSim(**env_hparams)
    env = ObsNormWrapper(env)

    # Policy
    policy_hparam = dict(
        # obs_layer=FNN(input_size=env.obs_space.flat_dim,
        #               output_size=env.act_space.flat_dim,
        #               hidden_sizes=[32, 32],
        #               hidden_nonlin=to.tanh),
        tau_init=1.,
        tau_learnable=True,
        capacity_learnable=False,
        output_nonlin=to.tanh,
        potentials_dyn_fcn=pd_capacity_21,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=10,
        batch_size=512,
        standardize_adv=False,
        standardizer=None,
        max_grad_norm=5.,
        lr=5e-4,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=20*env.max_steps,
        num_epoch=5,
        eps_clip=0.15,
        batch_size=512,
        max_grad_norm=5.,
        lr=3e-4,
        num_sampler_envs=8,
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
