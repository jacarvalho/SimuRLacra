"""
Script to plot the training progress
"""
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt

from pyrado.plotting.live_update import LiveFigureManager
from pyrado.plotting.curve import render_mean_std, render_lo_up_avg
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import read_csv_w_replace
from pyrado.utils.input_output import print_cbt

if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()
    file = osp.join(ex_dir, 'progress.csv')

    # Create plot manager that loads the grogress data from the CSV into a Pandas data frame called df
    lfm = LiveFigureManager(file, read_csv_w_replace, args, update_interval=5)


    @lfm.figure('Number of Rollouts')
    def num_rollouts(fig, df, args):
        if 'num_rollouts' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key num_rollouts in the data frame. Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.num_rollouts)), df.num_rollouts)
        plt.xlabel('iteration')
        plt.ylabel('number of rollouts')


    @lfm.figure('Average StepSequence Length')
    def avg_rollout_len(fig, df, args):
        if 'avg_rollout_len' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key avg_rollout_len in the data frame. Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.avg_rollout_len)), df.avg_rollout_len)
        plt.xlabel('iteration')
        plt.ylabel('rollout length')


    @lfm.figure('Return -- Average & Minimum & Maximum')
    def return_min_max_avg(fig, df, args):
        if 'avg_return' not in df.columns or 'min_return' not in df.columns or 'max_return' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key avg_return, min_return or max_return in the data frame.'
                          'Skipped the associated plot.')
            return False
        render_lo_up_avg(fig.gca(), np.arange(len(df.min_return)), df.min_return, df.max_return, df.avg_return,
                         x_label='iteration', y_label='return', curve_label='average')

    @lfm.figure('Return -- Average & Standard Deviation')
    def return_min_max_avg(fig, df, args):
        if 'avg_return' not in df.columns or 'std_return' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key avg_return or std_return in the data frame.'
                          'Skipped the associated plot.')
            return False
        render_mean_std(fig.gca(), np.arange(len(df.avg_return)), df.avg_return, df.std_return,
                        x_label='iteration', y_label='return', curve_label='average')
        plt.legend(loc='lower right')


    @lfm.figure('Return -- Average & Median (& Current)')
    def return_avg_median(fig, df, args):
        if 'avg_return' not in df.columns or 'median_return' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key avg_return or median_return in the data frame.'
                          'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.avg_return)), df.avg_return, label='average')
        plt.plot(np.arange(len(df.median_return)), df.median_return, label='median')
        if 'curr_policy_return' in df.columns:
            # If the algorithm is a subclass of ParameterExploring
            plt.plot(np.arange(len(df.curr_policy_return)), df.curr_policy_return, label='current')
        else:
            if args.verbose:
                print_cbt('Did not find the key curr_policy_return in the data frame. Skipped the associated plot.')
        plt.xlabel('iteration')
        plt.ylabel('return')
        plt.legend(loc='lower right')


    @lfm.figure('Explained Variance (R^2)')
    def explained_variance(fig, df, args):
        if 'explained_var' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key explained_var in the data frame. Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.explained_var)), df.explained_var)
        plt.xlabel('iteration')
        plt.ylabel('explained variance')
        plt.ylim(-1.1, 1.1)


    @lfm.figure("Exploration Strategy's Standard Deviation")
    def explstrat_std(fig, df, args):
        if 'avg_expl_strat_std' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key avg_expl_strat_std in the data frame. Skipped the associated plot.')
            return False
        # df.expl_strat_std = df.expl_strat_std.apply(lambda x: np.array(x))
        plt.plot(np.arange(len(df.avg_expl_strat_std)), df.avg_expl_strat_std, label='average')
        if 'min_expl_strat_std' in df.columns and 'max_expl_strat_std' in df.columns:
            plt.plot(np.arange(len(df.min_expl_strat_std)), df.min_expl_strat_std, label='smallest')
            plt.plot(np.arange(len(df.max_expl_strat_std)), df.max_expl_strat_std, label='largest')
        plt.xlabel('iteration')
        plt.ylabel('exploration std')
        plt.legend(loc='best')


    @lfm.figure("Exploration Strategy's Entropy")
    def explstrat_entropy(fig, df, args):
        if 'expl_strat_entropy' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key expl_strat_entropy in the data frame. Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.expl_strat_entropy)), df.expl_strat_entropy)
        plt.xlabel('iteration')
        plt.ylabel('exploration entropy')


    @lfm.figure("Average KL Divergence")
    def kl_divergence(fig, df, args):
        if 'avg_KL_old_new_' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key avg_KL_old_new_ in the data frame. Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.avg_KL_old_new_)), df.avg_KL_old_new_)
        plt.xlabel('iteration')
        plt.ylabel('KL divergence')


    @lfm.figure('Smallest and Largest Magnitude Policy Parameter')
    def extreme_policy_params(fig, df, args):
        if 'min_mag_policy_param' not in df.columns or 'max_mag_policy_param' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key min_mag_policy_param or max_mag_policy_param in the data'
                          'frame. Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.min_mag_policy_param)), df.min_mag_policy_param, label='smallest')
        plt.plot(np.arange(len(df.max_mag_policy_param)), df.max_mag_policy_param, label='largest')
        plt.xlabel('iteration')
        plt.ylabel('parameter value')
        plt.legend(loc='best')


    @lfm.figure('Loss Before and After Update Step')
    def loss_before_after(fig, df, args):
        if 'loss_before' not in df.columns or 'loss_after' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key loss_before or loss_after in the data frame.'
                          'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.loss_before)), df.loss_before, label='before')
        plt.plot(np.arange(len(df.loss_after)), df.loss_after, label='after')
        plt.xlabel('iteration')
        plt.ylabel('loss value')
        plt.legend(loc='best')


    @lfm.figure('Policy and Value Function Gradient L-2 Norm')
    def avg_grad_norm(fig, df, args):
        if 'avg_policy_grad_norm' not in df.columns or 'avg_V_fcn_grad_norm' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key avg_policy_grad_norm or avg_V_fcn_grad_norm in the data frame.'
                          'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.avg_policy_grad_norm)), df.avg_policy_grad_norm, label='policy')
        plt.plot(np.arange(len(df.avg_V_fcn_grad_norm)), df.avg_V_fcn_grad_norm, label='V-fcn')
        plt.xlabel('iteration')
        plt.ylabel('gradient norm')
        plt.legend(loc='best')


    """ CVaR Sampler """


    @lfm.figure('Full Average StepSequence Length')
    def full_avg_rollout_len(fig, df, args):
        if 'full_avg_rollout_len' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key avg_policy_grad_norm or avg_V_fcn_grad_norm in the data frame.'
                          'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.full_avg_rollout_len)), df.full_avg_rollout_len)
        plt.xlabel('iteration')
        plt.ylabel('rollout length')


    @lfm.figure('Full Return -- Average & Minimum & Maximum')
    def full_return_min_max_avg(fig, df, args):
        if 'full_avg_return' not in df.columns or 'full_min_return' not in df.columns or \
           'full_max_return' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key full_avg_return, full_min_return or full_max_return in the data frame.'
                          'Skipped the associated plot.')
            return False
        render_lo_up_avg(fig.gca(), np.arange(len(df.full_min_return)),
                         df.full_min_return, df.full_max_return, df.full_avg_return,
                         x_label='iteration', y_label='return', curve_label='average')
        plt.legend(loc='lower right')


    @lfm.figure('Full Return -- Average & Median & Standard Deviation')
    def return_avg_median_std(fig, df, args):
        if 'full_avg_return' not in df.columns or 'full_median_return' not in df.columns or \
           'full_std_return' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key full_avg_return, full_median_return or full_std_return in the data'
                          'frame. Skipped the associated plot.')
            return False
        render_mean_std(fig.gca(), np.arange(len(df.full_avg_return)), df.full_avg_return, df.full_std_return,
                        x_label='iteration', y_label='full return', curve_label='average')
        plt.plot(np.arange(len(df.full_median_return)), df.full_median_return, label='median')
        plt.legend(loc='lower right')


    """ REPS """


    @lfm.figure('REPS Dual Parameter')
    def eta(fig, df, args):
        if 'eta' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key eta in the data frame.' 'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.eta)), df.eta)
        plt.xlabel('iteration')
        plt.ylabel('$\eta$')


    @lfm.figure('Dual Loss Before and After Update Step')
    def loss_before_after(fig, df, args):
        if 'dual_loss_before' not in df.columns or 'dual_loss_after' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key dual_loss_before or dual_loss_after in the data frame.'
                          'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.dual_loss_before)), df.dual_loss_before, label='before')
        plt.plot(np.arange(len(df.dual_loss_after)), df.dual_loss_after, label='after')
        plt.xlabel('iteration')
        plt.ylabel('loss value')
        plt.legend(loc='best')


    """ SAC """


    @lfm.figure('SAC Temperature Parameter')
    def eta(fig, df, args):
        if 'alpha' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key alpha in the data frame.' 'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.alpha)), df.alpha)
        plt.xlabel('iteration')
        plt.ylabel(r'$\alpha$')


    @lfm.figure('Q-function Losses')
    def loss_before_after(fig, df, args):
        if 'Q1_loss' not in df.columns or 'Q2_loss' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key Q1_loss or Q2_loss in the data frame.'
                          'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.Q1_loss)), df.Q1_loss, label='$Q_1$')
        plt.plot(np.arange(len(df.Q2_loss)), df.Q2_loss, label='$Q_2$')
        plt.xlabel('iteration')
        plt.ylabel('loss value')
        plt.legend(loc='best')


    @lfm.figure('Policy Loss')
    def loss_before_after(fig, df, args):
        if 'policy_loss' not in df.columns:
            if args.verbose:
                print_cbt('Did not find the key policy_loss in the data frame.' 'Skipped the associated plot.')
            return False
        plt.plot(np.arange(len(df.policy_loss)), df.policy_loss)
        plt.xlabel('iteration')
        plt.ylabel('loss value')

    # Start update loop
    lfm.spin()
