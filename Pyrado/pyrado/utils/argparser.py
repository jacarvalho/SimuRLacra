import argparse


def get_argparser() -> argparse.ArgumentParser:
    """ Return Pyrado's default argument parser. """

    parser = argparse.ArgumentParser(description="Pyrado's default argument parser")

    parser.add_argument('--animation', dest='animation', action='store_true',
                        help="show a rendered animation (default: True)")
    parser.add_argument('--no_animation', dest='animation', action='store_false')
    parser.set_defaults(animation=True)

    parser.add_argument('--env_name', type=str, nargs='?',
                        help="name of the environment to use (e.g. 'qbb' or 'qcp-st')")

    parser.add_argument('--dt', type=float, default=1/500.,
                        help="environments time step size in seconds (default: 1/500)")

    parser.add_argument('--idcs', nargs='+', type=int, default=[0, 1],
                        help="list of indices casted to integer (default: [0, 1])")

    parser.add_argument('--iter', type=int, default=-1,
                        help="iteration to select for evaluation (default: -1 for last iteration)")

    parser.add_argument('--load_all', action='store_true', default=False,
                        help="load all quantities e.g. policies (default: False)")

    parser.add_argument('-d', '--ex_dir', type=str, nargs='?',
                        help="path to the experiment directory to load from")

    parser.add_argument('--max_steps', type=int, default=float('inf'),
                        help="maximum number of time steps to execute the environment")

    parser.add_argument('-m', '--mode', type=str, nargs='?',
                        help="general argument to specify different modes of various scripts (e.g. '2D')")

    parser.add_argument('-n', '--num_ro_per_config', type=int, default=180,
                        help="number of rollouts per environment configuration / domain parameter set (default: 120)")

    parser.add_argument('--num_envs', type=int, default=8,
                        help="number of environments to sample from in parallel (default: 8)")

    parser.add_argument('--num_runs', type=int, default=1,
                        help="number of runs for the overall experiment (default: 1)")

    parser.add_argument('--num_samples', type=int,
                        help="number of samples")

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', default=False,
                        help="display minimal information, the opposite of verbose (default: False)")

    parser.add_argument('--random_init_state', action='store_true', default=False,
                        help="use a random initial state (default: False)")

    parser.add_argument('--relentless', action='store_true', default=False,
                        help="don't stop (e.g. continue simulating after done flag was raised)")

    parser.add_argument('--remove_dr_wrappers', action='store_true', default=False,
                        help="remove all domain randomization wrappers (default: False)")

    parser.add_argument('--save_name', type=str, default='policy_conv',
                        help="name for the converted module, saved as <name>.pt (default: 'policy_conv')")

    parser.add_argument('-s', '--save_figures', action='store_true', default=False,
                        help="save all generated figures (default: False)")

    parser.add_argument('--seed', type=int, default=1001,
                        help='seed for the random number generators (default: 1001)')

    parser.add_argument('--use_tex', action='store_true', default=False,
                        help="use LaTeX fonts for plotting text with matplotlib (default: False)")

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                        help="display additional information (default: False)")

    parser.add_argument('--warmstart', dest='warmstart', action='store_true',
                        help="start a procedure with initialized parameters (e.g. for the policy")
    parser.add_argument('--from_scratch', dest='warmstart', action='store_false',
                        help="the opposite of 'warmstart'")

    return parser
