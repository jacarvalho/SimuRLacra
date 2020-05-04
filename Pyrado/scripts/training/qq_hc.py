"""
Train an agent to solve the Quanser Qube environment using Hill Climbing and an energy-based swing-up controller.
"""
from pyrado.algorithms.hc import HCNormal
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSim.name, HCNormal.name, 'ectrl', seed=1001)

    # Environment
    env_hparams = dict(dt=1/500., max_steps=4000)
    env = QQubeSim(**env_hparams)

    # Policy
    policy_hparam = dict(energy_gain=0.587, ref_energy=0.827)
    policy = QQubeSwingUpAndBalanceCtrl(env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        pop_size=10*policy.num_param,
        expl_factor=1.1,
        num_rollouts=12,
        expl_std_init=0.5,
        num_sampler_envs=12,
    )
    algo = HCNormal(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    algo.train(snapshot_mode='best')
