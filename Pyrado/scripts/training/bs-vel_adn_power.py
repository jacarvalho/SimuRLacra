"""
Train an agent to solve the Box Shelving task using Activation Dynamics Networks and P.
"""
import torch as to

from pyrado.algorithms.power import PoWER
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.box_shelving import BoxShelvingVelMPsSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic
from pyrado.policies.fnn import FNN

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, 'adn-power', 'posctrl_lin_obsnorm', seed=1001)
    # ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, 'adn-power', 'trqctrl_lin_obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxShelving_posCtrl.xml',  # gBoxShelving_posCtrl.xml or gBoxShelving_trqCtrl.xml
        dt=1/50.,
        max_steps=1500,
        fixed_init_state=True,
        bidirectional_mps=True,
        ref_frame='world',  # world, box, or upperGoal
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        collisionConfig={'file': 'collisionModel.xml'},
        taskCombinationMethod='mean',
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeVelocities=False,
        observeForceTorque=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=True,
        observeManipulabilityIndex=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
        observeDSGoalDistance=False,
    )
    env = BoxShelvingVelMPsSim(**env_hparams)
    env = ObsNormWrapper(env)

    # Policy
    policy_hparam = dict(
        # obs_layer=FNN(input_size=env.obs_space.flat_dim,
        #               output_size=env.act_space.flat_dim,
        #               hidden_sizes=[32, 16],
        #               hidden_nonlin=to.tanh),
        tau_init=5.,
        tau_learnable=True,
        output_nonlin=[to.tanh, to.tanh, to.tanh, to.tanh, to.tanh, to.sigmoid],
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        pop_size=policy.num_param,
        num_rollouts=20,
        num_is_samples=policy.num_param//2,
        expl_std_init=1.0,
        symm_sampling=False,
        num_sampler_envs=16,
    )
    algo = PoWER(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    print(env.obs_space)
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
