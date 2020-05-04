"""
Script to plot the GP's posterior after a Bayesian Domain Randomization experiment
"""
import os.path as osp
import torch as to
from matplotlib import pyplot as plt

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.gaussian_process import render_singletask_gp
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    cands = to.load(osp.join(ex_dir, 'candidates.pt'))
    cands_values = to.load(osp.join(ex_dir, 'candidates_values.pt')).unsqueeze(1)

    dim_cand = cands.shape[1]  # number of domain distribution parameters
    if dim_cand%2 != 0:
        raise pyrado.ShapeErr(msg='The dimension of domain distribution parameters must be a multiple of 2!')

    # Select dimensions to plot (ignored for 1D mode)
    if len(args.idcs) != 2:
        raise pyrado.ShapeErr(msg='Select exactly 2 indices!')

    fig_size = (12, 10)

    # Plot
    if args.mode == '1D':
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        widths = [1, 1]  # columnwise
        heights = [1]*(dim_cand//2)  # rowwise
        spec = fig.add_gridspec(nrows=dim_cand//2, ncols=2, width_ratios=widths, height_ratios=heights)

        for idx_r, row in enumerate(range(dim_cand//2)):
            for idx_c, col in enumerate(range(2)):  # 1st col means, 2nd col stds
                ax = fig.add_subplot(spec[row, col])
                render_singletask_gp(
                    ax, cands, cands_values, min_gp_obsnoise=1e-5,
                    idcs_sel=[idx_r*2 + idx_c], x_label='$\phi_{' + f'{idx_r*2 + idx_c}' + '}$',
                    y_label=r'$\hat{J}^{\textrm{real}}$', curve_label='mean', num_stds=2,
                    show_legend_posterior=False, show_legend_std=True, show_legend_data=args.verbose
                )

    elif args.mode == '2D':
        fig_mean = plt.figure(figsize=fig_size, constrained_layout=True)
        spec = fig_mean.add_gridspec(nrows=1, ncols=2, width_ratios=[12, 1], height_ratios=[1])
        ax_hm_mean = fig_mean.add_subplot(spec[0, 0])
        ax_cb_mean = fig_mean.add_subplot(spec[0, 1])

        fig_std = plt.figure(figsize=fig_size, constrained_layout=True)
        spec = fig_std.add_gridspec(nrows=1, ncols=2, width_ratios=[12, 1], height_ratios=[1])
        ax_hm_std = fig_std.add_subplot(spec[0, 0])
        ax_cb_std = fig_std.add_subplot(spec[0, 1])

        render_singletask_gp(
            [ax_hm_mean, ax_cb_mean, ax_hm_std, ax_cb_std], cands, cands_values, min_gp_obsnoise=1e-5,
            idcs_sel=args.idcs, x_label=f'$\phi_{args.idcs[0]}$', y_label=f'$\phi_{args.idcs[1]}$',
            z_label=r'$\hat{J}^{\textrm{real}}$', num_stds=2, resolution=51,
            show_legend_posterior=True, show_legend_std=True, show_legend_data=args.verbose, render_3D=False
        )

    elif args.mode == '3D':
        fig, ax = plt.subplots(1, subplot_kw={'projection': '3d'}, constrained_layout=True)
        render_singletask_gp(
            ax, cands, cands_values, min_gp_obsnoise=1e-5,
            idcs_sel=args.idcs, x_label=f'$\phi_{args.idcs[0]}$', y_label=f'$\phi_{args.idcs[1]}$',
            z_label=r'$\hat{J}^{\textrm{real}}$', num_stds=2, resolution=51,
            show_legend_posterior=False, show_legend_std=True, show_legend_data=args.verbose, render_3D=True
        )

    else:
        raise pyrado.ValueErr(given=args, eq_constraint="'1D', '2D', '3D'")

    if args.save_figures:
        if args.mode in ['1D', '3D']:
            fig.savefig(osp.join(ex_dir, f'gp_posterior_ret.pdf'), dpi=500)
            fig.savefig(osp.join(ex_dir, f'gp_posterior_ret.png'), dpi=500)
        elif args.mode == '2D':
            fig_mean.savefig(osp.join(ex_dir, f'gp_posterior_ret_mean.pdf'), dpi=500)
            fig_mean.savefig(osp.join(ex_dir, f'gp_posterior_ret_mean.png'), dpi=500)
            fig_std.savefig(osp.join(ex_dir, f'gp_posterior_ret_std.pdf'), dpi=500)
            fig_std.savefig(osp.join(ex_dir, f'gp_posterior_ret_std.png'), dpi=500)

    plt.show()
