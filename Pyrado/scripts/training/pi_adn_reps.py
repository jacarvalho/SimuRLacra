"""
Train an agent to solve the Planar Insert task using Activation Dynamics Networks and Relative Entropy Search.
"""
import torch as to

from pyrado.algorithms.reps import REPS
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.planar_insert import PlanarInsertSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PlanarInsertSim.name, 'adn-reps', '', seed=1001)
    # ex_dir = setup_experiment(PlanarInsertSim.name, 'adn-reps', 'obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gPlanarInsert6Link.xml',
        dt=1/50.,
        max_steps=1000,
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

    # Policy
    policy_hparam = dict(
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
        eps=0.1,
        gamma=0.999,
        pop_size=policy.num_param*20,
        num_rollouts=1,
        expl_std_init=2.0,
        expl_std_min=0.05,
        num_sampler_envs=16,
        num_epoch_dual=200,
        grad_free_optim=True,
        lr_dual=5e-4,
    )
    algo = REPS(ex_dir, env, policy, **algo_hparam)

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
