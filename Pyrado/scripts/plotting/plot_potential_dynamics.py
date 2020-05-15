"""
Script to plot dynamical systems used in Activation Dynamic Networks (ADN)
"""
import os
import numpy as np
import torch as to
import pandas as pd
import os.path as osp
from matplotlib import pyplot as plt

import pyrado
from pyrado.policies.adn import pd_linear, pd_cubic, pd_capacity_21_abs, pd_capacity_21, pd_capacity_32, \
    pd_capacity_32_abs
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt

if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Define the configurations to plot
    pd_fcn = pd_cubic  # function determining the potential dynamics
    p_min, p_max, num_p = -6., 6., 501
    print_cbt(f'Evaluating {pd_fcn.__name__} between {p_min} and {p_max}.', 'c')

    p = to.linspace(p_min, p_max, num_p)  # endpoint included
    nominal = dict(tau=1, s=0, c=4, kappa=0.1)
    nominals = dict(tau=nominal['tau']*to.ones(num_p),
                    s=nominal['s']*to.ones(num_p),
                    c=nominal['c']*to.ones(num_p),
                    kappa=nominal['kappa']*to.ones(num_p))
    config = dict()
    config['tau'] = to.linspace(0.2, 1.2, 6).repeat(num_p, 1).t()  # must include nominal value
    config['s'] = to.linspace(-2., 2., 5).repeat(num_p, 1).t()  # must include nominal value
    config['c'] = to.linspace(3., 5., 5).repeat(num_p, 1).t()  # must include nominal value
    config['kappa'] = to.linspace(0., 0.5, 5).repeat(num_p, 1).t()  # must include nominal value
    df_tau = pd.DataFrame(columns=['p_dot', 'p', 's', 'tau', 'c', 'kappa'])
    df_s = pd.DataFrame(columns=['p_dot', 'p', 's', 'tau', 'c', 'kappa'])
    df_c = pd.DataFrame(columns=['p_dot', 'p', 's', 'tau', 'c', 'kappa'])
    df_kappa = pd.DataFrame(columns=['p_dot', 'p', 's', 'tau', 'c', 'kappa'])

    if args.save_figures:
        save_dir = osp.join(pyrado.EVAL_DIR, 'dynamical_systems')
        os.makedirs(osp.dirname(save_dir), exist_ok=True)

    # Get the derivatives
    for tau in config['tau']:
        p_dot = pd_fcn(p=p, tau=tau, s=nominals['s'], capacity=nominals['c'], kappa=nominals['kappa'])
        df_tau = pd.concat([df_tau, pd.DataFrame(
            dict(p_dot=p_dot, p=p, s=nominals['s'], tau=tau, c=nominals['c'], kappa=nominals['kappa'])
        )], axis=0)

    for s in config['s']:
        p_dot = pd_fcn(p=p, s=s, tau=nominals['tau'], capacity=nominals['c'], kappa=nominals['kappa'])
        df_s = pd.concat([df_s, pd.DataFrame(
            dict(p_dot=p_dot, p=p, s=s, tau=nominals['tau'], c=nominals['c'], kappa=nominals['kappa'])
        )], axis=0)

    for c in config['c']:
        p_dot = pd_fcn(p=p, capacity=c, s=nominals['s'], tau=nominals['tau'], kappa=nominals['kappa'])
        df_c = pd.concat([df_c, pd.DataFrame(
            dict(p_dot=p_dot, p=p, c=c, s=nominals['s'], tau=nominals['tau'], kappa=nominals['kappa'])
        )], axis=0)

    for kappa in config['kappa']:
        p_dot = pd_fcn(p=p, kappa=kappa, s=nominals['s'], tau=nominals['tau'], capacity=nominals['c'])
        df_kappa = pd.concat([df_kappa, pd.DataFrame(
            dict(p_dot=p_dot, p=p, kappa=kappa, s=nominals['s'], tau=nominals['tau'], c=nominals['c'])
        )], axis=0)

    ''' tau '''
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the time constant tau: s = {nominal['s']}, c = {nominal['c']}, kappa = {nominal['kappa']}"
    )
    for tau in config['tau']:
        ax.plot(df_tau.loc[
                    (df_tau['s'] == nominal['s']) & (df_tau['tau'] == tau[0].numpy()) &
                    (df_tau['c'] == nominal['c']) & (df_tau['kappa'] == nominal['kappa'])
                    ]['p'],
                df_tau.loc[
                    (df_tau['s'] == nominal['s']) & (df_tau['tau'] == tau[0].numpy()) &
                    (df_tau['c'] == nominal['c']) & (df_tau['kappa'] == nominal['kappa'])
                    ]['p_dot'],
                label=f'$\\tau = {np.round(tau[0].numpy(), 2):.1f}$')

    # # Set fix points manually
    # ax.plot([-5, 5], [0, 0], marker='o', markerfacecolor='black', markeredgecolor='black', linestyle='None')
    # ax.plot([0], [0], marker='o', markerfacecolor='None', markeredgecolor='black', linestyle='None')

    ax.set_xlabel('$p$')
    ax.set_ylabel('$\dot{p}$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=3, fancybox=True)
    ax.grid()

    # Save
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(save_dir, f'potdyn-tau.{fmt}'), dpi=500)

    ''' s '''
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the stimulus s: tau = {nominal['tau']}, c = {nominal['c']}, kappa = {nominal['kappa']}"
    )
    for s in config['s']:
        ax.plot(df_s.loc[
                    (df_s['s'] == s[0].numpy()) & (df_s['tau'] == nominal['tau']) &
                    (df_s['c'] == nominal['c']) & (df_s['kappa'] == nominal['kappa'])
                    ]['p'],
                df_s.loc[
                    (df_s['s'] == s[0].numpy()) & (df_s['tau'] == nominal['tau']) &
                    (df_s['c'] == nominal['c']) & (df_s['kappa'] == nominal['kappa'])
                    ]['p_dot'],
                label=f'$s = {np.round(s[0].numpy(), 2):.1f}$')

    # # Approximately detect fix points
    # ax.plot(df_s.loc[(df_s['s'] == 0) & (df_s['tau'] == 1) & (abs(df_s['p_dot']) < 5e-2)]['p'],
    #         np.zeros(len(df_s.loc[(df_s['s'] == 0) & (df_s['tau'] == 1) & (abs(df_s['p_dot']) < 5e-2)]['p'])),
    #         marker='o', markerfacecolor='None', markeredgecolor='k', linestyle='None')

    ax.set_xlabel('$p$')
    ax.set_ylabel('$\dot{p}$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=5, fancybox=True)
    ax.grid()

    # Save
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(save_dir, f'potdyn-s.{fmt}'), dpi=500)

    ''' c '''
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the capacity C: tau = {nominal['tau']}, s = {nominal['s']}, kappa = {nominal['kappa']}"
    )
    for c in config['c']:
        ax.plot(df_c.loc[
                    (df_c['c'] == c[0].numpy()) & (df_c['tau'] == nominal['tau']) &
                    (df_c['s'] == nominal['s']) & (df_c['kappa'] == nominal['kappa'])
                    ]['p'],
                df_c.loc[
                    (df_c['c'] == c[0].numpy()) & (df_c['tau'] == nominal['tau']) &
                    (df_c['s'] == nominal['s']) & (df_c['kappa'] == nominal['kappa'])
                    ]['p_dot'],
                label=f'$C = {np.round(c[0].numpy(), 2):.1f}$')

    # # Approximately detect fix points
    # ax.plot(df_c.loc[(df_c['c'] == 5) & (df_c['tau'] == 1) & (abs(df_c['p_dot']) < 5e-2)]['p'],
    #         np.zeros(len(df_c.loc[(df_s['c'] == 5) & (df_c['tau'] == 1) & (abs(df_c['p_dot']) < 5e-2)]['p'])),
    #         marker='o', markerfacecolor='None', markeredgecolor='k', linestyle='None')

    ax.set_xlabel('$p$')
    ax.set_ylabel('$\dot{p}$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=3, fancybox=True)
    ax.grid()

    # Save
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(save_dir, f'potdyn-C.{fmt}'), dpi=500)

    ''' kappa '''
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the decay factor kappa: tau = {nominal['tau']}, s = {nominal['s']}, c = {nominal['c']}"
    )
    for kappa in config['kappa']:
        ax.plot(df_kappa.loc[
                    (df_kappa['kappa'] == kappa[0].numpy()) & (df_kappa['tau'] == nominal['tau']) &
                    (df_kappa['s'] == nominal['s']) & (df_kappa['c'] == nominal['c'])
                    ]['p'],
                df_kappa.loc[
                    (df_kappa['kappa'] == kappa[0].numpy()) & (df_kappa['tau'] == nominal['tau']) &
                    (df_kappa['s'] == nominal['s']) & (df_kappa['c'] == nominal['c'])
                    ]['p_dot'],
                label=f'$\\kappa = {np.round(kappa[0].numpy(), 2):.1f}$')

    ax.set_xlabel('$p$')
    ax.set_ylabel('$\dot{p}$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=3, fancybox=True)
    ax.grid()

    # Save
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(save_dir, f'potdyn-kappa.{fmt}'), dpi=500)

    plt.show()
