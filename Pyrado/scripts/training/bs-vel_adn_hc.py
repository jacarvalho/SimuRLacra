"""
Train an agent to solve the Box Shelving task task using Activation Dynamics Networks and Hill Climbing.
"""
import torch as to

from pyrado.algorithms.hc import HCNormal
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.box_shelving import BoxShelvingVelMPsSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic, pd_capacity_21_abs
from pyrado.policies.fnn import FNN


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, f'adn-{HCNormal.name}', 'posctrl_lin_obsnorm', seed=1001)
    # ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, f'adn-{HCNormal.name}', 'trqctrl_lin_obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxShelving_trqCtrl.xml',  # gBoxShelving_posCtrl.xml or gBoxShelving_trqCtrl.xml
        dt=1/100.,
        max_steps=2000,
        fixed_init_state=True,
        bidirectional_mps=True,
        ref_frame='world',  # world, box, or upperGoal
        mps_left=None,  # use defaults
        collisionConfig={'file': 'collisionModel.xml'},
        taskCombinationMethod='sum',
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeVelocities=False,
        observeForceTorque=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=False,
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
        # init_param_kwargs=dict(sigmoid_nlin=True),
        potentials_dyn_fcn=pd_cubic,  # pd_cubic, pd_capacity_21_abs
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        pop_size=policy.num_param//2,
        expl_factor=1.1,
        num_rollouts=1,
        expl_std_init=1.0,
        num_sampler_envs=32,
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
    print(env.obs_space)
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
