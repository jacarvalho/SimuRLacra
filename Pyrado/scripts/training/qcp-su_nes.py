"""
Train an agent to solve the Quanser Cart-Pole swing-up task using Natural Evolution Strategies.
"""
from pyrado.algorithms.nes import NES
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.policies.rnn import LSTMPolicy, GRUPolicy

if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QCartPoleSwingUpSim.name, NES.name, 'gru_short', seed=1001)

    # Environments
    env_hparams = dict(dt=1/500., max_steps=6000, long=False)
    env = QCartPoleSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        hidden_size=16,
        num_recurrent_layers=1,
        # init_param_kwargs=dict(t_max=50)
    )
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=10000,
        pop_size=None,
        num_rollouts=12,
        eta_mean=1.,
        eta_std=None,
        expl_std_init=1.0,
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
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
