"""
Run a PD-controller with the parameter from Quanser on the real device.
By default all controllers in this script run infinitely.
"""
import torch as to

import pyrado
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleSwingUpReal, QCartPoleStabReal
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.policies.environment_specific import QBallBalancerPDCtrl, QCartPoleSwingUpAndBalanceCtrl,\
    QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


# python ... -env-name qcp-st

if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Set up PD-controller
    if args.env_name in QBallBalancerReal.name:
        env = QBallBalancerReal(args.dt, args.max_steps)
        policy = QBallBalancerPDCtrl(env.spec, kp=to.diag(to.tensor([3.45, 3.45])), kd=to.diag(to.tensor([2.11, 2.11])))
        print_cbt('Set up controller for the QBallBalancerReal environment.', 'c')

    elif args.env_name == QCartPoleStabReal.name:
        env = QCartPoleStabReal(args.dt, args.max_steps)
        policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)
        print_cbt('Set up controller for the QCartPoleStabReal environment.', 'c')

    elif args.env_name == QCartPoleSwingUpReal.name:
        env = QCartPoleSwingUpReal(args.dt, args.max_steps)
        policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)
        print_cbt('Set up controller for the QCartPoleSwingUpReal environment.', 'c')

    elif args.env_name == QQubeReal.name:
        env = QQubeReal(args.dt, args.max_steps)
        policy = QQubeSwingUpAndBalanceCtrl(env.spec)
        print_cbt('Set up controller for the QQubeReal environment.', 'c')


    else:
        raise pyrado.ValueErr(given=args.env_name,
                              eq_constraint=f'{QBallBalancerReal.name}, {QCartPoleSwingUpReal.name}, '
                                            f'{QCartPoleStabReal.name}, or {QQubeReal.name}')

    # Run on device
    done = False
    while not done:
        print_cbt('Running predefined controller ...', 'c', bright=True)
        ro = rollout(env, policy, eval=True, render_mode=RenderMode(text=args.verbose))
        done, _, _ = after_rollout_query(env, policy, ro)
