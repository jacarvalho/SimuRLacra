"""
Train an agent to solve the PlanarInsert task using Activation Dynamics Networks and Hill Climbing.
"""
import torch as to

from pyrado.algorithms.hc import HCNormal
from pyrado.domain_randomization.default_randomizers import get_default_randomizer
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
    ex_dir = setup_experiment(PlanarInsertSim.name, f'adn-{HCNormal.name}', 'obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gPlanarInsert6Link.xml',
        dt=1/50.,
        max_steps=800,
        # max_dist_force=1e1,
        taskCombinationMethod='sum',  # 'sum', 'mean',  'product', or 'softmax'
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeForceTorque=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeGoalDistance=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
        # usePhysicsNode=True,
    )
    env = PlanarInsertSim(**env_hparams)
    env = ObsNormWrapper(env)

    # randomizer = get_default_randomizer(env)
    # env = ActDelayWrapper(env)
    # randomizer.add_domain_params(UniformDomainParam(name='act_delay', mean=2, halfspan=2, clip_lo=0, roundint=True))
    # env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(
        # obs_layer=FNN(input_size=env.obs_space.flat_dim,
        #               output_size=env.act_space.flat_dim,
        #               hidden_sizes=[8, 8],
        #               hidden_nonlin=to.tanh,
        #               dropout=0.),
        tau_init=5.,
        tau_learnable=True,
        kappa_init=0.02,
        kappa_learnable=True,
        output_nonlin=to.sigmoid,
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        pop_size=policy.num_param,
        expl_factor=1.1,
        num_rollouts=1,
        expl_std_init=2.,
        num_sampler_envs=1,
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
    print(env.act_space.labels)
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
