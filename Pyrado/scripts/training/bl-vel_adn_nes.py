"""
Train an agent to solve the Box Lifting task using Activation Dynamics Networks and Natural Evolutionary Strategies.
"""
import torch as to

from pyrado.algorithms.nes import NES
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.box_lifting import BoxLiftingVelMPsSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic, pd_capacity_21_abs
from pyrado.policies.fnn import FNN
from pyrado.policies.rnn import LSTMPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxLiftingVelMPsSim.name, f'adn-{NES.name}', 'posctrl_lin_obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxLifting_posCtrl.xml',  # gBoxLifting_posCtrl.xml or gBoxLifting_trqCtrl.xml
        dt=1/100.,
        max_steps=2000,
        fixed_init_state=True,
        ref_frame='basket',  # world, basket, or box
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        collisionConfig={'file': 'collisionModel.xml'},
        taskCombinationMethod='sum',
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeForceTorque=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=False,
        observeDSGoalDistance=False,
    )
    env = BoxLiftingVelMPsSim(**env_hparams)
    env = ObsNormWrapper(env)

    # Policy
    policy_hparam = dict(
        # obs_layer=FNN(input_size=env.obs_space.flat_dim,
        #               output_size=env.act_space.flat_dim,
        #               hidden_sizes=[32, 32],
        #               hidden_nonlin=to.tanh),
        tau_init=5.,
        tau_learnable=True,
        output_nonlin=to.tanh,
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)
    # policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)

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
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
