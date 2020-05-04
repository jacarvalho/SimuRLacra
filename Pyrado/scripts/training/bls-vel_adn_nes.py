"""
Train an agent to solve the simplified Box Lifting task using Activation Dynamics Networks and
Natural Evolutionary Strategies.
"""
import torch as to

from pyrado.algorithms.nes import NES
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.domain_randomization.domain_parameter import UniformDomainParam, NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.box_lifting import BoxLiftingSimpleVelMPsSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic, pd_capacity_21_abs


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxLiftingSimpleVelMPsSim.name, f'adn-{NES.name}', 'lin', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxLiftingSimple_trqCtrl.xml',  # gBoxLiftingSimple_posCtrl.xml or gBoxLiftingSimple_trqCtrl.xml
        dt=1/50.,
        max_steps=1000,
        ref_frame='box',  # world, basket, or box
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        collisionConfig=None,
        taskCombinationMethod='sum',
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeVelocities=False,
        observeForceTorque=True,
        observeCollisionCost=False,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
        observeDSGoalDistance=False,
    )
    env = BoxLiftingSimpleVelMPsSim(**env_hparams)
    # env = ObsNormWrapper(env)
    randomizer = DomainRandomizer(
        NormalDomainParam(name='box_width', mean=0.18, std=0.02),
        # NormalDomainParam(name='box_mass', mean=0.5, std=0.05),
        # UniformDomainParam(name='box_friction_coefficient', mean=1.4, halfspan=0.2),
        # UniformDomainParam(name='basket_friction_coefficient', mean=0.8, halfspan=0.1),
    )
    # env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(
        tau_init=2.,
        tau_learnable=True,
        output_nonlin=to.tanh,
        potentials_dyn_fcn=pd_cubic,
        init_param_kwargs=dict(sigmoid_nlin=True),
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=5000,
        pop_size=None,
        num_rollouts=1,
        eta_mean=1.,
        eta_std=None,
        expl_std_init=1.0,
        symm_sampling=False,
        transform_returns=True,
        num_sampler_envs=4,
    )
    algo = NES(ex_dir, env, policy, **algo_hparam)

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
