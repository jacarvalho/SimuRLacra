"""
Train an agent to solve the Planar-3-Link task using Neural Fields and Hill Climbing.
"""
import torch as to

from pyrado.algorithms.cem import CEM
from pyrado.algorithms.hc import HCNormal
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.environments.rcspysim.planar_3_link import Planar3LinkIKSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.neural_fields import NFPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(Planar3LinkIKSim.name, HCNormal.name, NFPolicy.name, seed=101)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        dt=1/50.,
        max_steps=800,
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
        observeTaskSpaceDiscrepancy=True,
    )
    env = Planar3LinkIKSim(**env_hparams)
    eub = {
        'GD_DS0': 2.,
        'GD_DS1': 2.,
        'GD_DS2': 2.,
    }
    env = ObsNormWrapper(env, explicit_ub=eub)
    env = ObsPartialWrapper(env, idcs=['Effector_Xd', 'Effector_Zd'])

    # Policy
    policy_hparam = dict(
        hidden_size=5,
        conv_out_channels=1,
        mirrored_conv_weights=True,
        conv_kernel_size=3,
        conv_padding_mode='circular',
        init_param_kwargs=dict(bell=True),
        activation_nonlin=to.sigmoid,
        tau_init=1e-1,
        tau_learnable=True,
        kappa_init=None,
        kappa_learnable=True,
        potential_init_learnable=True,
    )
    policy = NFPolicy(spec=env.spec, dt=env.dt, **policy_hparam)
    # policy.param_values = to.load('/home/muratore/Software/SimuRLacra/Pyrado/data/temp/p3l-ik/hc/2020-06-30_15-06-57--nf/policy.pt').param_values
    print(policy)

    # Algorithm
    # algo_hparam = dict(
    #     max_iter=50,
    #     pop_size=policy.num_param,
    #     num_rollouts=1,
    #     num_is_samples=policy.num_param//10,
    #     expl_std_init=1.0,
    #     expl_std_min=0.02,
    #     extra_expl_std_init=0.5,
    #     extra_expl_decay_iter=5,
    #     full_cov=False,
    #     symm_sampling=False,
    #     num_sampler_envs=6,
    # )
    # algo = CEM(ex_dir, env, policy, **algo_hparam)
    algo_hparam = dict(
        max_iter=100,
        pop_size=policy.num_param,
        expl_factor=1.05,
        num_rollouts=1,
        expl_std_init=0.5,
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
