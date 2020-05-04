"""
Script to plot the results from the 2D domain parameter grid evaluations of a single policy.
"""
import os
import os.path as osp
import pandas as pd
from matplotlib import colors

from matplotlib import pyplot as plt
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.heatmap import render_heatmap
from pyrado.plotting.utils import AccNorm
from pyrado.utils.argparser import get_argparser


def _plot_and_save(
        df: pd.DataFrame,
        index: str,
        column: str,
        index_label: str,
        column_label: str,
        values: str = 'ret',
        add_sep_colorbar: bool = True,
        norm: colors.Normalize = None,
        save_figure: bool = False,
        save_dir: str = None
):
    if index in df.columns and column in df.columns:
        # Pivot table (by default averages over identical index / columns cells)
        df_pivot = df.pivot_table(index=index, columns=column, values=values)

        # Generate the plot
        fig_hm, fig_cb = render_heatmap(df_pivot, add_sep_colorbar=add_sep_colorbar, norm=norm,
                                        ylabel=index_label, xlabel=column_label)

        # Save heat map and color bar if desired
        if save_figure:
            name = '-'.join([index, column])
            fig_hm.savefig(osp.join(save_dir, f'hm-{name}.pdf'))
            if fig_cb is not None:
                fig_cb.savefig(osp.join(save_dir, f'cb-{name}.pdf'))


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Commonly scale the colorbars of all plots
    accnorm = AccNorm()

    # Get the experiment's directory to load from
    exp_dir = ask_for_experiment()
    eval_parent_dir = osp.join(exp_dir, 'eval_domain_grid')
    assert osp.isdir(eval_parent_dir)

    if args.load_all:
        list_eval_dirs = [tmp[0] for tmp in os.walk(eval_parent_dir)][1:]
    else:
        list_eval_dirs = [
                osp.join(eval_parent_dir, 'FILL_IN'),
        ]

    # Loop over all evaluations
    for eval_dir in list_eval_dirs:
        assert osp.isdir(eval_dir)

        # Load the data
        df = pd.read_pickle(osp.join(eval_dir, 'df_sp_grid_nd.pkl'))

        # Remove constant rows
        df = df.loc[:, df.apply(pd.Series.nunique) != 1]

        ''' QBallBalancerSim '''

        _plot_and_save(df, 'm_ball', 'r_ball', r'$m_{\mathrm{ball}}$', r'$r_{\mathrm{ball}}$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

        _plot_and_save(df, 'g', 'r_ball', '$g$', r'$r_{\mathrm{ball}}$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

        _plot_and_save(df, 'J_l', 'J_m', '$J_l$', '$J_m$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

        _plot_and_save(df, 'eta_g', 'eta_m', r'$\eta_g$', r'$\eta_m$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

        _plot_and_save(df, 'k_m', 'R_m', '$k_m$', '$R_m$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

        _plot_and_save(df, 'B_eq', 'c_frict', r'$B_{\mathrm{eq}}$', r'$c_{\mathrm{frict}}$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

        _plot_and_save(df, 'V_thold_x_pos', 'V_thold_x_neg', r'$V_{\mathrm{thold,x-}}$', r'$V_{\mathrm{thold,x+}}$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

        _plot_and_save(df, 'V_thold_y_pos', 'V_thold_y_neg', r'$V_{\mathrm{thold,y-}}$', r'$V_{\mathrm{thold,y+}}$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

        _plot_and_save(df, 'm_ball', 'act_delay', r'$m_{\mathrm{ball}}$', r'$a_{\mathrm{delay}}$',
                       add_sep_colorbar=True, norm=accnorm, save_figure=args.save_figures, save_dir=eval_dir)

    plt.show()
