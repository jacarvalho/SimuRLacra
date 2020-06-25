"""
Train an agent to solve the Box Shelving task using Activation Dynamics Networks and Natural Evolutionary Strategies.
"""
import torch as to

from pyrado.algorithms.nes import NES
from pyrado.domain_randomization.default_randomizers import get_default_randomizer
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environment_wrappers.observation_normalization import ObsRunningNormWrapper, ObsNormWrapper
from pyrado.environments.rcspysim.box_shelving import BoxShelvingVelMPsSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic, pd_capacity_21_abs
from pyrado.policies.fnn import FNN
from pyrado.policies.rnn import LSTMPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    # ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, f'adn-{NES.name}', 'posctrl_lin_obsnorm', seed=1001)
    ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, f'adn-{NES.name}', 'posctrl_fnn', seed=1001)
    # ex_dir = setup_experiment(BoxShelvingVelMPsSim.name, NES.name, 'posctrl_lstm_obsnorm', seed=1001)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBoxShelving_posCtrl.xml',  # gBoxShelving_posCtrl.xml or gBoxShelving_trqCtrl.xml
        dt=1/100.,
        max_steps=2000,
        fix_init_state=True,
        bidirectional_mps=True,
        ref_frame='world',  # world, box, or upperGoal
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
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
    # randomizer = get_default_randomizer(env)
    # env = ActDelayWrapper(env)
    # randomizer.add_domain_params(UniformDomainParam(name='act_delay', mean=4, halfspan=4, clip_lo=0, roundint=True))
    # env = DomainRandWrapperLive(env, randomizer)
    # env = ObsRunningNormWrapper(env)
    # env = ObsNormWrapper(env)

    # Policy
    policy_hparam = dict(
        obs_layer=FNN(input_size=env.obs_space.flat_dim,
                      output_size=env.act_space.flat_dim,
                      hidden_sizes=[32, 32],
                      hidden_nonlin=to.tanh),
        tau_init=5.,
        tau_learnable=True,
        activation_nonlin=[to.tanh, to.tanh, to.tanh, to.tanh, to.tanh, to.sigmoid],
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)
    # policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
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
