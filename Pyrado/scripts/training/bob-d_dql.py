"""
Train an agent to solve the discrete Ball-on-Beam environment using Deep Q-Leaning.

.. note::
    The hyper-parameters are not tuned at all!
"""
import torch as to

from pyrado.algorithms.dql import DQL
from pyrado.environments.pysim.ball_on_beam import BallOnBeamDiscSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import DiscrActQValFNNPolicy

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamDiscSim.name, DQL.name, 'lin-ident-sin', seed=1001)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=500)
    env = BallOnBeamDiscSim(**env_hparams)

    # Policy
    policy_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    policy = DiscrActQValFNNPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=5000,
        memory_size=100*env.max_steps,
        eps_init=0.1286,
        eps_schedule_gamma=0.9955,
        gamma=0.995,
        target_update_intvl=5,
        num_batch_updates=10,
        max_grad_norm=0.5,
        min_steps=1,
        batch_size=256,
        num_sampler_envs=4,
        lr=7.461e-4,
    )
    algo = DQL(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
