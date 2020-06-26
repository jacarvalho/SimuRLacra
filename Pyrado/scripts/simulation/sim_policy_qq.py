"""
Simulate (with animation) a rollout in an environment.
"""
import pprint

import pyrado
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.data_types import RenderMode


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Load the environment and the policy
    env = QQubeSim(args.dt, args.max_steps)  # runs infinitely by default
    policy = QQubeSwingUpAndBalanceCtrl(env.spec)
    print_cbt('Set up controller for the QQubeSim environment.', 'c')

    # Override the time step size if specified
    if args.dt is not None:
        env.dt = args.dt


    if args.remove_dr_wrappers:
        env = remove_all_dr_wrappers(env, verbose=True)

    # Use the environments number of steps in case of the default argument (inf)
    max_steps = env.max_steps if args.max_steps == pyrado.inf else args.max_steps


    # Simulate
    done, state, param = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=args.verbose, video=args.animation),
                     eval=True, max_steps=max_steps, stop_on_done=not args.relentless,
                     reset_kwargs=dict(domain_param=param, init_state=state))
        print_domain_params(env.domain_param)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
    pyrado.close_vpython()
