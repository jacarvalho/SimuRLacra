"""
Script to get the maximizer of a GP's posterior mean given saved data from a BayRn experiment
"""
import os.path as osp
import torch as to

from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.math import UnitCubeProjector


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load the required data
    cands = to.load(osp.join(ex_dir, 'candidates.pt'))
    cands_values = to.load(osp.join(ex_dir, 'candidates_values.pt')).unsqueeze(1)
    bounds = to.load(osp.join(ex_dir, 'bounds.pt'))
    uc_normalizer = UnitCubeProjector(bounds[0, :], bounds[1, :])

    # Compute and print the argmax
    BayRn.argmax_posterior_mean(cands, cands_values, uc_normalizer, num_restarts=500, num_samples=1000)
