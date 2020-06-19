"""
Train an agent to solve the Box Shelving task task using Activation Dynamics Networks and Cross-Entropy Method.
"""
import torch as to

from pyrado.algorithms.cem import CEM
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.mujoco.openai_hopper import HopperSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.neural_fields import NFPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(HopperSim.name, f'nf-{CEM.name}', 'lin', seed=1001)

    # Environment
    env_hparams = dict()
    env = HopperSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        hidden_size=20,
        conv_out_channels=1,
        conv_kernel_size=9,
        conv_padding_mode='reflected',
        activation_nonlin=to.tanh,
        tau_init=1.,
        tau_learnable=True,
    )
    policy = NFPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        pop_size=200,
        num_rollouts=4,
        num_is_samples=10,
        expl_std_init=1.,
        expl_std_min=0.02,
        extra_expl_std_init=1.,
        extra_expl_decay_iter=20,
        full_cov=False,
        symm_sampling=False,
        num_sampler_envs=8,
    )
    algo = CEM(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
