"""
Train an agent to solve the Box Shelving task task using Activation Dynamics Networks and Hill Climbing.
"""
import torch as to

from pyrado.algorithms.hc import HCNormal
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.mujoco.openai_hopper import HopperSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.adn import ADNPolicy, pd_cubic, pd_capacity_21_abs
from pyrado.policies.fnn import FNN

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(HopperSim.name, f'adn-{HCNormal.name}', 'lin', seed=1001)

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
        potentials_dyn_fcn=pd_cubic,
        scaling_layer=False,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        pop_size=10*policy.num_param,
        expl_factor=1.05,
        num_rollouts=4,
        expl_std_init=2.,
        num_sampler_envs=1,
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
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
