"""
Train an agent to solve the OpenAI Gym task using Activation Dynamics Networks and Hill Climbing.
"""
import torch as to

from pyrado.algorithms.nes import NES
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.openai_classical_control import GymEnv
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_capacity_21


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(GymEnv.name, f'adn', 'actnorm', seed=101)

    # env_hparams = dict(env_name='Pendulum-v0')
    # env = GymEnv(**env_hparams)

    env_hparams = dict(dt=1/100., max_steps=800)
    env = BallOnBeamSim(**env_hparams)

    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        dt=env.dt,
        tau_init=env.dt,  # 1/env.dt requires large policy params for fast reactions
        tau_learnable=False,
        capacity_learnable=False,
        output_nonlin=to.tanh,
        potentials_dyn_fcn=pd_capacity_21,
        scaling_layer=False,
        # init_param_kwargs=dict(uniform_bias=True)
    )
    policy = ADNPolicy(spec=env.spec, **policy_hparam)

    # policy_hparam = dict(hidden_size=1, num_recurrent_layers=1, init_param_kwargs=dict(t_max=800))  # LSTM
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)

    # policy_hparam = dict(
    #     feats=FeatureStack([identity_feat, sin_feat, cos_feat])
    # )
    # policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
        pop_size=None,
        num_rollouts=12,
        eta_mean=2.,
        eta_std=None,
        expl_std_init=10.0,
        symm_sampling=False,
        transform_returns=True,
        num_sampler_envs=12,
    )
    algo = NES(ex_dir, env, policy, **algo_hparam)

    # algo_hparam = dict(
    #     max_iter=200,
    #     pop_size=policy.num_param,
    #     expl_factor=1.1,
    #     num_rollouts=12,
    #     expl_std_init=1.0,
    #     num_sampler_envs=12,
    # )
    # algo = HCNormal(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
