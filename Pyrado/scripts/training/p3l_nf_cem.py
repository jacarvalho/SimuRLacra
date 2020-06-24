"""
Train an agent to solve the Planar-3-Link task using Neural Fields and Hill Climbing.
"""
import torch as to

from pyrado.algorithms.cem import CEM
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.planar_3_link import Planar3LinkIKSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.neural_fields import NFPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(Planar3LinkIKSim.name, f'nf-{CEM.name}', '', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        dt=1/50.,
        max_steps=1000,
        max_dist_force=None,
        taskCombinationMethod='sum',
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeVelocities=False,
        observeForceTorque=True,
        observeCollisionCost=False,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeGoalDistance=True,
        observeDynamicalSystemDiscrepancy=False,
    )
    env = Planar3LinkIKSim(**env_hparams)
    eub = {
        'GD_DS0': 2.,
        'GD_DS1': 2.,
        'GD_DS2': 2.,
    }
    env = ObsNormWrapper(env, explicit_ub=eub)

    # Policy
    policy_hparam = dict(
        hidden_size=5,
        conv_out_channels=1,
        # conv_kernel_size=5,
        conv_padding_mode='circular',
        activation_nonlin=to.sigmoid,
        tau_init=1e-1,
        tau_learnable=True,
    )
    policy = NFPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=2*policy.num_param,
        num_rollouts=2,
        num_is_samples=2*policy.num_param//10,
        expl_std_init=0.5,
        expl_std_min=0.02,
        extra_expl_std_init=0.5,
        extra_expl_decay_iter=10,
        full_cov=False,
        symm_sampling=False,
        num_sampler_envs=6,
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
