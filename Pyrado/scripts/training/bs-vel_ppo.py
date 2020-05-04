"""
Train an agent to solve the Box Shelving task using Proximal Policy Optimization.
"""
import torch as to

from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.box_shelving import BoxShelvingVelMPsSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import RNNPolicy, LSTMPolicy, GRUPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    # ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, PPO.name, 'fnn_obsnorm', seed=1001)
    # ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, PPO.name, 'fnn', seed=1001)
    ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, PPO.name, 'trqCtrl_lstm_obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxShelving_posCtrl.xml',  # gBoxShelving_posCtrl.xml or gBoxShelving_trqCtrl.xml
        dt=1/50.,
        max_steps=1000,
        fixed_init_state=True,
        bidirectional_mps=True,
        ref_frame='world',  # world, box, or upperGoal
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        collisionConfig={'file': 'collisionModel.xml'},
        taskCombinationMethod='mean',
        checkJointLimits=True,
        collisionAvoidanceIK=False,
        observeVelocities=False,
        observeForceTorque=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
    )
    env = BoxShelvingVelMPsSim(**env_hparams)
    env = ObsNormWrapper(env)
    # env = ObsNormWrapper(env, explicit_lb={'ManipIdx': 0.}, explicit_ub={'ManipIdx': 100.})
    # env = ActNormWrapper(env)

    # Policy
    # policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)  # FNN
    # policy_hparam = dict(hidden_size=32, num_recurrent_layers=1, hidden_nonlin='tanh')  # RNN
    policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    # policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = RNNPolicy(spec=env.spec, **policy_hparam)
    policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    # policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    # value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    # value_fcn_hparam = dict(hidden_siz=32, num_recurrent_layers=1, hidden_nonlin='tanh')  # RNN
    value_fcn_hparam = dict(hidden_size=16, num_recurrent_layers=1)  # LSTM & GRU
    # value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = RNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    value_fcn = LSTMPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.999,
        lamda=0.985,
        num_epoch=6,
        batch_size=100,
        lr=4e-4,
        standardize_adv=False,
        max_grad_norm=5,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
        min_steps=30*env.max_steps,
        num_epoch=2,
        eps_clip=0.075,
        batch_size=100,
        std_init=0.1,
        lr=6e-4,
        num_sampler_envs=32,
        max_grad_norm=5,
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed, snapshot_mode='best')
