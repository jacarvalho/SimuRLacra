"""
Script to get the maximizer of a GP's posterior given saved data from a BayRn experiment
"""
import os.path as osp
import torch as to

from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.bayrn import BayRn
from pyrado.algorithms.cem import CEM
from pyrado.algorithms.nes import NES
from pyrado.algorithms.power import PoWER
from pyrado.algorithms.ppo import PPO, PPO2
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.math import UnitCubeProjector


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load the environment and the policy
    env_sim, policy, kwout = load_experiment(ex_dir, args)

    # Load the required data
    cands = to.load(osp.join(ex_dir, 'candidates.pt'))
    cands_values = to.load(osp.join(ex_dir, 'candidates_values.pt')).unsqueeze(1)
    bounds = to.load(osp.join(ex_dir, 'bounds.pt'))
    uc_normalizer = UnitCubeProjector(bounds[0, :], bounds[1, :])

    # Decide on which algorithm to use via the mode argument
    if args.mode == PPO.name:
        critic = GAE(kwout['value_fcn'], **kwout['hparams']['critic'])
        subroutine = PPO(ex_dir, env_sim, policy, critic, **kwout['hparams']['subroutine'])
    elif args.mode == PPO2.name:
        critic = GAE(kwout['value_fcn'], **kwout['hparams']['critic'])
        subroutine = PPO2(ex_dir, env_sim, policy, critic, **kwout['hparams']['subroutine'])
    elif args.mode == CEM.name:
        subroutine = CEM(ex_dir, env_sim, policy, **kwout['hparams']['subroutine'])
    elif args.mode == NES.name:
        subroutine = NES(ex_dir, env_sim, policy, **kwout['hparams']['subroutine'])
    elif args.mode == PoWER.name:
        subroutine = PoWER(ex_dir, env_sim, policy, **kwout['hparams']['subroutine'])
    else:
        raise NotImplementedError('Only PPO, PPO2, CEM, NES, and PoWER are implemented so far.')

    if args.warmstart:
        ppi = policy.param_values.data
        vpi = kwout['value_fcn'].param_values.data
    else:
        ppi = None
        vpi = None

    # Train the policy on the most lucrative domain
    BayRn.train_argmax_policy(ex_dir, env_sim, subroutine, num_restarts=500, num_samples=1000,
                              policy_param_init=ppi, valuefcn_param_init=vpi)
