"""
Train an agent to solve the Ball-on-Beam environment using Soft Actor-Critic.
"""
import torch as to

from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.policies.two_headed import TwoHeadedFNNPolicy, TwoHeadedGRUPolicy
from pyrado.utils.data_types import EnvSpec
from pyrado.algorithms.sac import SAC
from pyrado.spaces import ValueFunctionSpace, BoxSpace
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, SAC.name, 'fnn', seed=1001)

    # Environment
    env_hparams = dict(dt=1/50., max_steps=300)
    env = BallOnBeamSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    # policy_hparam = dict(
    #     shared_hidden_sizes=[32, 32],
    #     shared_hidden_nonlin=to.relu,
    # )
    # policy = TwoHeadedFNNPolicy(spec=env.spec, **policy_hparam)
    policy_hparam = dict(
        shared_hidden_size=8,
        shared_num_recurrent_layers=1,
    )
    policy = TwoHeadedGRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    q_fcn_hparam = dict(
        hidden_sizes=[32, 32],
        hidden_nonlin=to.relu
    )
    obsact_space = BoxSpace.cat([env.obs_space, env.act_space])
    q_fcn_1 = FNNPolicy(spec=EnvSpec(obsact_space, ValueFunctionSpace), **q_fcn_hparam)
    q_fcn_2 = FNNPolicy(spec=EnvSpec(obsact_space, ValueFunctionSpace), **q_fcn_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000*env.max_steps,
        memory_size=100*env.max_steps,
        gamma=0.99,
        num_batch_updates=1,
        tau=0.99,  # 0.995
        alpha_init=0.2,  # 0.2
        learn_alpha=False,
        target_update_intvl=1,
        standardize_rew=False,
        min_steps=1,
        batch_size=256,
        num_sampler_envs=4,
        lr=3e-4,
    )
    algo = SAC(ex_dir, env, policy, q_fcn_1, q_fcn_2, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(q_fcn=q_fcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
