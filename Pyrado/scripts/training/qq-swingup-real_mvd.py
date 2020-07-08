"""
Train an agent to solve the Quanser Qube environment using Episodic Measure-Valued Derivatives.
"""
from pyrado.algorithms.emvd import EMVD
from pyrado.algorithms.torchdistributions import GaussianDiagonalLogStdParametrization, GaussianDiagonal
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environments.quanser.quanser_qube import QQubeReal
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
    ex_dir = setup_experiment(QQubeReal.name + 'swinguponly', EMVD.name, QQubeSwingUpAndBalanceCtrl.name, seed=0)

    # Environment
    env_hparams = dict(dt=1/500., max_steps=5000)
    env = QQubeReal(**env_hparams)
    # env = ActNormWrapper(env)

    # Search distribution
    init_loc = np.array([np.log(0.02),
                         np.log(50.),
                         0.3],
                        dtype=np.float64)
    init_std = 0.5 * np.ones(init_loc.shape[0], dtype=np.float64)


    dist = GaussianDiagonalLogStdParametrization(init_loc=init_loc, init_std=init_std)

    # Policy
    policy_hparam = dict(
        pd_gains = to.tensor([-1.7313308, 35.976177, -1.58682, 3.0102878])
    )
    policy = QQubeSwingUpAndBalanceCtrl(env.spec, **policy_hparam)

    # Set the policy parameters to the initial ones...
    # policy.param_values = to.tensor(init_loc)

    # Sample a policy from the final search distribution
    policy.param_values = to.tensor(init_loc)


    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=1,
        num_rollouts=1,
        expl_std_init=1.0,
        expl_std_min=0.0,
        num_sampler_envs=1,
        n_mc_samples_gradient=1,
        coupling=True,
        lr=5e-2,
        optim='Adam',
        real_env=True
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
