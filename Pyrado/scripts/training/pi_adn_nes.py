"""
Train an agent to solve the PlanarInsert task using Activation Dynamics Networks and Natural Evolution Strategies.
"""
import torch as to

from pyrado.algorithms.nes import NES
from pyrado.domain_randomization.default_randomizers import get_default_randomizer, get_empty_randomizer
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.planar_insert import PlanarInsertSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic
from pyrado.policies.fnn import FNN

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PlanarInsertSim.name, f'adn-{NES.name}', 'fnn_obsnorm_actdelay-4', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Vortex',  # Bullet or Vortex
        graphFileName='gPlanarInsert6Link.xml',
        dt=1/50.,
        max_steps=800,
        # max_dist_force=1e1,
        taskCombinationMethod='sum',  # 'sum', 'mean',  'product', or 'softmax'
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeForceTorque=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeGoalDistance=False,
        observeDynamicalSystemDiscrepancy=True,
        observeTaskSpaceDiscrepancy=True,
        usePhysicsNode=True,
    )
    env = PlanarInsertSim(**env_hparams)
    # Explicit normalization bounds
    elb = {
        'DiscrepDS_Effector_X': -1.,
        'DiscrepDS_Effector_Z': -1.,
        'DiscrepDS_Effector_Bd': -1,
    }
    eub = {
        'DiscrepDS_Effector_X': 1.,
        'DiscrepDS_Effector_Z': 1.,
        'DiscrepDS_Effector_Bd': 1,
    }
    env = ObsNormWrapper(env, explicit_lb=elb, explicit_ub=eub)

    randomizer = get_default_randomizer(env)
    # randomizer = get_empty_randomizer()
    env = ActDelayWrapper(env)
    randomizer.add_domain_params(UniformDomainParam(name='act_delay', mean=2, halfspan=2, clip_lo=0, roundint=True))
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(
        obs_layer=FNN(input_size=env.obs_space.flat_dim,
                      output_size=env.act_space.flat_dim,
                      hidden_sizes=[32, 32],
                      hidden_nonlin=to.tanh,
                      dropout=0.),
        tau_init=5.,
        tau_learnable=True,
        kappa_init=0.02,
        kappa_learnable=True,
        activation_nonlin=to.sigmoid,
        potentials_dyn_fcn=pd_cubic,
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
        num_sampler_envs=12,
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
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
