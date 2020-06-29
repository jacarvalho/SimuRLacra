"""
Train an agent to solve the Box Shelving task task using Activation Dynamics Networks and Cross-Entropy Method.
"""
import numpy as np
import torch as to

from pyrado.algorithms.cem import CEM
from pyrado.algorithms.hc import HCNormal
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.neural_fields import NFPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(OneMassOscillatorSim.name, HCNormal.name, NFPolicy.name, seed=1001)

    # Environment
    env_hparams = dict(dt=1/50., max_steps=200)
    env = OneMassOscillatorSim(**env_hparams, task_args=dict(state_des=np.array([0.5, 0])))
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        hidden_size=3,
        conv_out_channels=1,
        mirrored_conv_weights=True,
        conv_kernel_size=3,
        conv_padding_mode='circular',
        init_param_kwargs=dict(bell=True),
        activation_nonlin=to.sigmoid,
        tau_init=1e-2,
        tau_learnable=False,
        potential_init_learnable=False,
    )
    policy = NFPolicy(spec=env.spec, dt=env.dt, **policy_hparam)
    print(policy)

    # Algorithm
    # algo_hparam = dict(
    #     max_iter=50,
    #     pop_size=100,
    #     num_rollouts=6,
    #     num_is_samples=10,
    #     expl_std_init=0.5,
    #     expl_std_min=0.02,
    #     extra_expl_std_init=0.5,
    #     extra_expl_decay_iter=10,
    #     full_cov=False,
    #     symm_sampling=False,
    #     num_sampler_envs=8,
    # )
    # algo = CEM(ex_dir, env, policy, **algo_hparam)
    algo_hparam = dict(
        max_iter=100,
        pop_size=4*policy.num_param,
        expl_factor=1.05,
        num_rollouts=6,
        expl_std_init=0.3,
        num_sampler_envs=6,
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
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
