"""
Train an agent to solve the Quanser Qube environment using Episodic Measure-Valued Derivatives.
"""
from pyrado.algorithms.emvd import EMVD
from pyrado.algorithms.torchdistributions import GaussianDiagonalLogStdParametrization
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.policies.features import FeatureStack, identity_feat, sign_feat, abs_feat, squared_feat, qubic_feat, \
    bell_feat, RandFourierFeat, MultFeat
from pyrado.policies.linear import LinearPolicy
import torch as to
import numpy as np


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    # ex_dir = setup_experiment(QQubeSim.name, PoWER.name, f'{LinearPolicy}_actnorm', seed=1)
    ex_dir = setup_experiment(QQubeSim.name, EMVD.name, QQubeSwingUpAndBalanceCtrl.name, seed=0)

    # Environment
    env_hparams = dict(dt=1/500., max_steps=5000)
    env = QQubeSim(**env_hparams)
    # env = ActNormWrapper(env)

    # Search distribution
    init_loc = np.array([np.log(0.02), np.log(50.),
                         -2., 35., -1.5, 3.],
                        dtype=np.float64)
    init_std = 1.0 * np.ones(init_loc.shape[0], dtype=np.float64)
    dist = GaussianDiagonalLogStdParametrization(init_loc=init_loc, init_std=init_std)

    # Policy
    policy_hparam = dict(
        ref_energy=init_loc[0],
        energy_gain=init_loc[1],
        energy_th_gain=0.3, # This parameter is fixed.
        acc_max=5.,
        alpha_max_pd_enable=10.,
        pd_gains=to.tensor([init_loc[2], init_loc[3], init_loc[4], init_loc[5]], dtype=to.float64)
    )
    policy = QQubeSwingUpAndBalanceCtrl(env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=150,
        pop_size=1,
        num_rollouts=1,
        expl_std_init=1.0,
        expl_std_min=0.0,
        num_sampler_envs=12,
        n_mc_samples_gradient=1,
        coupling=True,
        lr=1e-1,
        optim='Adam'
    )


    algo = EMVD(ex_dir, env, policy, dist, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed, snapshot_mode='best')
