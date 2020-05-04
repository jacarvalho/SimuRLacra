"""
Train an agent to solve the simplified Box Flipping task using Activation Dynamics Networks and Hill Climbing.
"""
import torch as to

from pyrado.algorithms.hc import HCNormal
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.box_flipping import BoxFlippingPosMPsSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic, pd_capacity_21_abs


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxFlippingPosMPsSim.name, f'adn-{HCNormal.name}', 'lin', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxFlipping_trqCtrl.xml',  # gBoxFlipping_posCtrl.xml or gBoxFlipping_trqCtrl.xml
        dt=1/100.,
        max_steps=2000,
        ref_frame='table',  # world, table, or box
        collisionConfig={'file': 'collisionModel.xml'},
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeVelocities=False,
        observeForceTorque=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=False,
        observeDSGoalDistance=False,
    )
    env = BoxFlippingPosMPsSim(**env_hparams)
    # env = ObsNormWrapper(env)

    # Policy
    policy_hparam = dict(
        tau_init=2.,
        tau_learnable=True,
        output_nonlin=to.tanh,
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=policy.num_param,
        expl_factor=1.05,
        num_rollouts=1,
        expl_std_init=1.,
        num_sampler_envs=12,
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
    print(env.obs_space.labels)
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
