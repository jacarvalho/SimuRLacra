"""
Train an agent to solve the Planar-3-Link task using Activation Dynamics Networks and Hill Climbing.
"""
import torch as to

import pyrado
from pyrado.algorithms.hc import HCNormal
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.planar_3_link import Planar3LinkIKSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import pd_cubic, ADNPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(Planar3LinkIKSim.name, HCNormal.name, ADNPolicy.name, seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        dt=1/50.,
        max_steps=700,
        task_args=dict(consider_velocities=True),
        max_dist_force=None,
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
        observeTaskSpaceDiscrepancy=False,
    )
    env = Planar3LinkIKSim(**env_hparams)
    env = ActNormWrapper(env)
    eub = {
        'GD_DS0': 2.,
        'GD_DS1': 2.,
        'GD_DS2': 2.,
    }
    env = ObsNormWrapper(env, explicit_ub=eub)
    print(env.act_space)
    print(env.obs_space)

    # Policy
    policy_hparam = dict(
        tau_init=1e-1,
        tau_learnable=True,
        kappa_init=1e-2,
        kappa_learnable=True,
        activation_nonlin=to.tanh,
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=10*policy.num_param,
        expl_factor=1.05,
        num_rollouts=1,
        expl_std_init=0.05,
        num_sampler_envs=6,
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
