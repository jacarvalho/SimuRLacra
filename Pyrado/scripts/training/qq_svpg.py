"""
Train agents to solve the Ball-on-Beam environment using Stein Variational Policy Gradient.
"""
import torch as to

from pyrado.algorithms.advantage import ValueFunctionSpace, GAE
from pyrado.algorithms.svpg import SVPG, SVPGParticle
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, SVPG.name, 'fnn_actnorm', seed=1)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=800)
    env = QQubeSim(**env_hparams)
    env = ActNormWrapper(env)

    # Specification of actor an critic (will be instantiated in SVPG)
    actor_hparam = dict(
        hidden_sizes=[64, 64],
        hidden_nonlin=to.relu,
    )
    value_fcn_hparam = dict(
        hidden_sizes=[16, 16],
        hidden_nonlin=to.tanh,
    )
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=5,
        lr=5e-4,
        standardize_adv=False,
        max_grad_norm=5.,
    )
    actor = FNNPolicy(spec=env.spec, **actor_hparam)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic = GAE(value_fcn, **critic_hparam)
    particle = SVPGParticle(env.spec, actor, critic)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=30*env.max_steps,
        num_particles=5,
        temperature=10,
        lr=5e-4,
        horizon=50,
        num_sampler_envs=10,
        serial=True
    )
    algo = SVPG(ex_dir, env, particle, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
