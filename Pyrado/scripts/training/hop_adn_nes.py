"""
Train an agent to solve the Half-Cheetah task using Activation Dynamics Networks and Natural Evolution Strategies.
"""
import torch as to

from pyrado.algorithms.nes import NES
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.mujoco.openai_hopper import HopperSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic, pd_capacity_21_abs, pd_capacity_32_abs
from pyrado.policies.fnn import FNN


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(HopperSim.name, NES.name, ADNPolicy.name, seed=1002)

    # Environment
    env_hparams = dict()
    env = HopperSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        # obs_layer=FNN(input_size=env.obs_space.flat_dim,
        #               output_size=env.act_space.flat_dim,
        #               hidden_sizes=[32, 32],
        #               hidden_nonlin=to.tanh),
        tau_init=1.,
        tau_learnable=True,
        kappa_learnable=True,
        capacity_learnable=False,
        activation_nonlin=to.tanh,
        potentials_dyn_fcn=pd_capacity_21_abs,
        scaling_layer=False,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=5000,
        pop_size=100,
        num_rollouts=4,
        eta_mean=2.,
        eta_std=None,
        expl_std_init=2.0,
        symm_sampling=False,
        transform_returns=True,
        num_sampler_envs=8,
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
    algo.train(seed=ex_dir.seed)
