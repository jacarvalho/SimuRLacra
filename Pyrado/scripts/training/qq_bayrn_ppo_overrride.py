"""
Learn the domain parameter distribution of masses and lengths of the Quanser Qube while using a handcrafted
randomization for the remaining domain parameters. Continue in the same directory of a previous experiment.
"""
import joblib
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import load_dict_from_yaml, ask_for_experiment


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = ask_for_experiment()

    # Environments
    hparams = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))
    env_sim = joblib.load(osp.join(ex_dir, 'env_sim.pkl'))
    env_real = joblib.load(osp.join(ex_dir, 'env_real.pkl'))

    # Policy
    policy = to.load(osp.join(ex_dir, 'policy.pt'))

    # Critic
    critic = to.load(osp.join(ex_dir, 'critic.pt'))

    # Subroutine
    algo_hparam = hparams['subroutine']
    # algo_hparam.update({'num_sampler_envs': 1})
    ppo = PPO(ex_dir, env_sim, policy, critic, **algo_hparam)

    # Set the boundaries for the GP
    bounds = to.load(osp.join(ex_dir, 'bounds.pt'))

    # Algorithm
    algo = BayRn(ex_dir, env_sim, env_real, subroutine=ppo, bounds=bounds, **hparams['algo'])

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=hparams['seed'], load_dir=ex_dir)
