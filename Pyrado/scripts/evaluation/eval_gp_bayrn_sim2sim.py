"""
Script for the paper plot the GP's posterior after a Bayesian Domain Randomization sim-to-sim experiment
"""
import numpy as np
import os.path as osp
import torch as to
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import pyrado
from matplotlib import pyplot as plt
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.gaussian_process import render_singletask_gp
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    cands = to.load(osp.join(ex_dir, 'candidates.pt'))
    cands_values = to.load(osp.join(ex_dir, 'candidates_values.pt')).unsqueeze(1)
    bounds = to.load(osp.join(ex_dir, 'bounds.pt'))

    dim_cand = cands.shape[1]  # number of domain distribution parameters
    if dim_cand%2 != 0:
        raise pyrado.ShapeErr(msg='The dimension of domain distribution parameters must be a multiple of 2!')

    # Select dimensions to plot (ignored for 1D mode)
    if len(args.idcs) != 2:
        raise pyrado.ShapeErr(msg='Select exactly 2 indices!')

    fig_size = (pyrado.figsize_IEEE_1col_square[0]*0.75, pyrado.figsize_IEEE_1col_square[0]*0.75)

    # Plot 2D
    fig_mean, ax_hm_mean = plt.subplots(1, figsize=fig_size, constrained_layout=True)
    _, ax_cb_mean = plt.subplots(1, figsize=fig_size, constrained_layout=True)

    fig_std, ax_hm_std = plt.subplots(1, figsize=fig_size, constrained_layout=True)
    _, ax_cb_std = plt.subplots(1, figsize=fig_size, constrained_layout=True)

    # Nice color map from seaborn
    # hm_cmap = sns.cubehelix_palette(light=.9, dark=.1, reverse=True, as_cmap=True)
    # hm_cmap = sns.light_palette("muted_navy", reverse=True, as_cmap=True)
    hm_cmap = ListedColormap(sns.color_palette("YlGnBu", n_colors=100)[::-1])
    # hm_cmap = ListedColormap(sns.color_palette("YlOrRd", n_colors=100)[::-1])
    # hm_cmap = ListedColormap(sns.color_palette("OrRd", n_colors=100)[::-1])
    # scat_cmap = LinearSegmentedColormap.from_list('white_to_gray', [(1., 1., 1.), (.4, .4, .4)], N=256)
    scat_cmap = LinearSegmentedColormap.from_list('light_to_dark_gray', [(.8, .8, .8), (.2, .2, .2)], N=256)

    render_singletask_gp(
        [ax_hm_mean, ax_cb_mean, ax_hm_std, ax_cb_std], cands, cands_values, min_gp_obsnoise=1e-5,
        # data_x_min=bounds[0, args.idcs], data_x_max=bounds[1, args.idcs],
        idcs_sel=args.idcs, x_label=f'$m_p$', y_label=f'$m_r$', heatmap_cmap=hm_cmap,
        z_label=r'$\hat{J}^{\textrm{real}}$', num_stds=2, resolution=151, legend_data_cmap=scat_cmap,
        show_legend_posterior=True, show_legend_std=True, show_legend_data=args.verbose, render_3D=False,
    )

    # ax_hm_mean.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax_hm_std.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Plot the ground truth domain parameter configuration
    ax_hm_mean.scatter(0.026, 0.097, c='firebrick', marker='o', s=60)  # forestgreen
    ax_hm_std.scatter(0.026, 0.097, c='firebrick', marker='o', s=60)  # forestgreen

    if args.save_figures:
        fig_mean.savefig(osp.join(ex_dir, f'gp_posterior_ret_mean.pdf'), dpi=500)
        fig_mean.savefig(osp.join(ex_dir, f'gp_posterior_ret_mean.pgf'), dpi=500)
        fig_std.savefig(osp.join(ex_dir, f'gp_posterior_ret_std.pdf'), dpi=500)
        fig_std.savefig(osp.join(ex_dir, f'gp_posterior_ret_std.pgf'), dpi=500)

    plt.show()
