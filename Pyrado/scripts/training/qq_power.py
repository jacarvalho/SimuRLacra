"""
Train an agent to solve the Quanser Qube environment using Policy learning by Weighting Exploration with the Returns.
"""
from pyrado.algorithms.power import PoWER
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.policies.features import FeatureStack, identity_feat, sign_feat, abs_feat, squared_feat, qubic_feat, \
    bell_feat, RandFourierFeat, MultFeat
from pyrado.policies.linear import LinearPolicy
import torch as to


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    # ex_dir = setup_experiment(QQubeSim.name, PoWER.name, f'{LinearPolicy}_actnorm', seed=1)
    ex_dir = setup_experiment(QQubeSim.name, PoWER.name, QQubeSwingUpAndBalanceCtrl.name, seed=1)

    # Environment
    env_hparams = dict(dt=1/500., max_steps=5000)
    env = QQubeSim(**env_hparams)
    # env = ActNormWrapper(env)

    # Policy
    # policy_hparam = dict(
    #     # feats=FeatureStack([RandFourierFeat(env.obs_space.flat_dim, num_feat=20, bandwidth=env.obs_space.bound_up)])
    #     feats=FeatureStack([identity_feat, sign_feat, abs_feat, squared_feat,
    #                         MultFeat([2, 5]), MultFeat([3, 5]), MultFeat([4, 5])])
    # )
    # policy = LinearPolicy(spec=env.spec, **policy_hparam)
    # policy_hparam = dict(energy_gain=0.587, ref_energy=0.827)
    policy_hparam = dict(
        ref_energy=0.02,
        energy_gain=50.,
        energy_th_gain=0.4,
        acc_max=5.,
        alpha_max_pd_enable=10.,
        pd_gains=to.tensor([-2, 35, -1.5, 3])
    )
    policy = QQubeSwingUpAndBalanceCtrl(env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=50,
        num_rollouts=10,
        num_is_samples=10,
        expl_std_init=0.5,
        expl_std_min=0.02,
        symm_sampling=False,
        num_sampler_envs=12,
    )
    algo = PoWER(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed, snapshot_mode='best')
