"""
Train an agent to solve the simplified Box Flipping task using Cross-Entorpy Method.
"""
from pyrado.algorithms.cem import CEM
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.environments.rcspysim.box_flipping import BoxFlippingPosMPsSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, const_feat
from pyrado.policies.linear import LinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxFlippingPosMPsSim.name, f'{CEM.name}', 'trqctrl_obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxFlipping_trqCtrl.xml',  # gBoxFlipping_posCtrl.xml or gBoxFlipping_trqCtrl.xml
        dt=1/100.,
        max_steps=2000,
        ref_frame='world',  # world, table, or box
        collisionConfig={'file': 'collisionModel.xml'},
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        checkJointLimits=True,
        collisionAvoidanceIK=False,
        observeManipulators=False,
        observeBoxOrientation=False,
        observeVelocities=True,
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
    # env = ObsPartialWrapper(env, idcs=['Box_Yd'])

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([const_feat, identity_feat])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=100,
        num_rollouts=1,
        num_is_samples=20,
        expl_std_init=2.,
        expl_std_min=0.02,
        extra_expl_std_init=2.,
        extra_expl_decay_iter=20,
        full_cov=False,
        symm_sampling=False,
        num_sampler_envs=8,
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
    print(env.obs_space)
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
