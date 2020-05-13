"""
Test predefined energy-based controller to make the Quanser Cart-Pole swing up or balancing task.
"""
import pyrado
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim, QCartPoleStabSim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.environment_specific import QCartPoleSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt

if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Set up environment and policy (swing-up works reliably if is sampling frequency is >= 400 Hz)
    if args.env_name == 'qcp-su':
        env = QCartPoleSwingUpSim(dt=args.dt, max_steps=int(10/args.dt), long=False)
        policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)

    elif args.env_name == 'qcp-st':
        env = QCartPoleStabSim(dt=args.dt, max_steps=int(4/args.dt), long=False)
        policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)

    else:
        raise pyrado.ValueErr(given=args.env_name, eq_constraint="'qcp-su' or 'qcp-st'")

    # Simulate
    done, param, state = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=False, video=True), eval=True,
                     reset_kwargs=dict(domain_param=param, init_state=state))
        print_domain_params(env.domain_param)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
