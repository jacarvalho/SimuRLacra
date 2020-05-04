"""
Script to evaluate the UCBOG and OG of multiple experiments
"""
import os
import os.path as osp
import pandas as pd

import pyrado
from matplotlib import pyplot as plt
from pyrado.logger.experiment import load_dict_from_yaml
from pyrado.plotting.curve import render_mean_std
from pyrado.sampling.sequences import *
from pyrado.utils.experiments import filter_los_by_lok


if __name__ == '__main__':
    save_name = 'FILL_IN'

    # Add every experiment with (partially) matching key
    filter_key = ['FILL_IN']

    # Get the experiments' directories to load from
    ex_dirs = []
    ex_dirs.extend([tmp[0] for tmp in os.walk(osp.join(pyrado.EXP_DIR, 'FILL_IN', 'FILL_IN'))][1:])
    ex_dirs = filter_los_by_lok(ex_dirs, filter_key)
    print(f'Number of loaded experiments: {len(ex_dirs)}')

    dfs = []
    for ex_dir in ex_dirs:
        dfs.append(pd.read_csv(osp.join(ex_dir, 'OG_log.csv')))
    df = pd.concat(dfs, axis=0)  # missing values are filled with nan

    # Compute metrics using pandas (nan values are ignored)
    print(f'Index counts\n{df.index.value_counts()}')

    # Compute metrics using pandas (nan values are ignored)
    UCBOG_mean = df.groupby(df.index)['UCBOG'].mean()
    UCBOG_std = df.groupby(df.index)['UCBOG'].std()
    Gn_mean = df.groupby(df.index)['Gn_est'].mean()
    Gn_std = df.groupby(df.index)['Gn_est'].std()
    rnd_mean = df.groupby(df.index)['ratio_neg_diffs'].mean()
    rnd_std = df.groupby(df.index)['ratio_neg_diffs'].std()

    # Reconstruct the number of domains per iteration
    nc = []
    for ex_dir in ex_dirs:
        hparam = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))
        nc_init = hparam['SPOTA']['nc_init']
        if hparam['SPOTA']['sequence_cand'] == 'sequence_add_init':
            nc.append(sequence_add_init(nc_init, len(UCBOG_mean) - 1)[1])
        elif hparam['SPOTA']['sequence_cand'] == 'sequence_rec_double':
            nc.append(sequence_rec_double(nc_init, len(UCBOG_mean) - 1)[1])
        elif hparam['SPOTA']['sequence_cand'] == 'sequence_rec_sqrt':
            nc.append(sequence_rec_sqrt(nc_init, len(UCBOG_mean) - 1)[1])
        else:
            raise pyrado.ValueErr(given=hparam['SPOTA']['sequence_cand'],
                                  eq_constraint="'sequence_add_init', 'sequence_rec_double', 'sequence_rec_sqrt'")
    nc_means = np.floor(np.mean(np.asarray(nc), axis=0))

    # Plots
    fig1, axs = plt.subplots(3, constrained_layout=True)
    render_mean_std(axs[0], UCBOG_mean.index, UCBOG_mean, UCBOG_std, '', '', 'UCBOG')
    render_mean_std(axs[1], Gn_mean.index, Gn_mean, Gn_std, '', '', '$\\hat{G}_n$')
    render_mean_std(axs[2], rnd_mean.index, rnd_mean, rnd_std, '', '', 'rnd')

    fig2, ax1 = plt.subplots(1, figsize=pyrado.figsize_IEEE_1col_18to10)
    fig2.canvas.set_window_title(f'Final UCBOG value: {UCBOG_mean.values[-1]}')
    render_mean_std(ax1, UCBOG_mean.index, UCBOG_mean, UCBOG_std, 'iteration', 'UCBOG', '', show_legend=False)
    ax1.set_ylabel('UCBOG', color='C0')
    ax2 = ax1.twinx()  # second y axis
    ax2.plot(nc_means, color='C1')
    ax2.set_ylabel('number of domains $n_c$', color='C1')
    fig2.savefig(osp.join(pyrado.EVAL_DIR, 'optimality_gap', 'ucbog_' + save_name + '.pdf'), dpi=500)
    plt.show()
