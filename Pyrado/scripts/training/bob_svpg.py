"""
Train agents to solve the Ball-on-Beam environment using Stein Variational Policy Gradient.
"""
import torch as to

from pyrado.algorithms.svpg import SVPG
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, SVPG.name, '', seed=1001)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=500)
    env = BallOnBeamSim(**env_hparams)
    env = ActNormWrapper(env)

    # Specification of actor an critic (will be instantiated in SVPG)
    actor_hparam = dict(
        hidden_sizes=[64],
        hidden_nonlin=to.relu,
    )
    value_fcn_hparam = dict(
        hidden_sizes=[32],
        hidden_nonlin=to.tanh,
    )
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=5,
        lr=1e-3,
        standardize_adv=False,
        max_grad_norm=5.,
    )
    particle_hparam = dict(actor=actor_hparam, value_fcn=value_fcn_hparam, critic=critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=200,
        min_steps=30*env.max_steps,
        num_particles=3,
        temperature=1,
        lr=1e-3,
        std_init=1.0,
        horizon=50,
        num_sampler_envs=12,
    )
    algo = SVPG(ex_dir, env, particle_hparam, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
