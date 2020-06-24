"""
Train an agent to solve the Half-Cheetah environment using Neural Fields and Cross-Entropy Method.
"""
import torch as to

from pyrado.algorithms.cem import CEM
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.mujoco.openai_half_cheetah import HalfCheetahSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.neural_fields import NFPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(HalfCheetahSim.name, f'nf-{CEM.name}', 'lin', seed=1001)

    # Environment
    env_hparams = dict()
    env = HalfCheetahSim()
    # env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        hidden_size=21,
        conv_out_channels=1,
        conv_kernel_size=5,
        conv_padding_mode='circular',
        activation_nonlin=to.sigmoid,
        mirrored_conv_weights=True,
        tau_init=1e-1,
        tau_learnable=True,
    )
    policy = NFPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=policy.num_param,
        num_rollouts=4,
        num_is_samples=policy.num_param//10,
        expl_std_init=1.,
        expl_std_min=0.02,
        extra_expl_std_init=1.,
        extra_expl_decay_iter=25,
        full_cov=False,
        symm_sampling=False,
        num_sampler_envs=32,
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
