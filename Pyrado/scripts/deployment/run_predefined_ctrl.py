"""
Run a PD-controller with the parameter from Quanser on the real device.
By default all controllers in this script run infinitely.
"""
import torch as to
import numpy as np

import pyrado
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleSwingUpReal, QCartPoleStabReal
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.policies.environment_specific import QBallBalancerPDCtrl, QCartPoleSwingUpAndBalanceCtrl,\
    QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode, EnvSpec
from pyrado.utils.input_output import print_cbt

from pyrado.policies.base import Policy


class CartpoleStabilizerPolicy(Policy):
    """ Swing-up and balancing controller for the Quanser Cart-Pole """

    def __init__(self,
                 env_spec: EnvSpec,
                 K: np.ndarray = np.array(
                     [1.2278416e+00, 4.5279346e+00, -1.2385756e-02, 6.0038762e+00, -4.1818547e+00]),
                 u_max: float = 18.,
                 v_max: float = 12.):
        """
        Constructor

        :param env_spec: environment specification
        :param u_max: maximum energy gain
        :param v_max: maximum voltage the control signal will be clipped to
        :param long: flag for long or short pole
        """
        super().__init__(env_spec)

        # Store inputs
        self.u_max = u_max
        self.v_max = v_max

        self.K_pd = to.tensor(K)

        self._max_u = 3.0

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Calculate the controller output.

        :param obs: observation from the environment
        :return act: controller output [V]
        """
        x, sin_th, cos_th, x_dot, theta_dot = obs

        act = self.K_pd.dot(obs)

        # Return the clipped action
        act = act.clamp(-self.v_max, self.v_max)

        # Denormalize action
        lb, ub = -self._max_u, self._max_u
        act = lb + (act + 1) * (ub - lb) / 2

        # Bound
        act = self._bound(act, -self._max_u, self._max_u)

        return act.view(1)  # such that when act is later converted to numpy it does not become a float

    @staticmethod
    def _bound(x, min_value, max_value):
        """
        Method used to bound state and action variables.

        Args:
            x: the variable to bound;
            min_value: the minimum value;
            max_value: the maximum value;

        Returns:
            The bounded variable.

        """
        return np.maximum(min_value, np.minimum(x, max_value))



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
        policy = CartpoleStabilizerPolicy(
            env.spec,
            K=np.array([1.2278416e+00, 4.5279346e+00, -1.2385756e-02, 6.0038762e+00, -4.1818547e+00])
        )
        # policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)
        print_cbt('Set up controller for the QCartPoleStabReal environment.', 'c')

    elif args.env_name == QCartPoleSwingUpReal.name:
        env = QCartPoleSwingUpReal(args.dt, args.max_steps)
        policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)
        print_cbt('Set up controller for the QCartPoleSwingUpReal environment.', 'c')

    elif args.env_name == QQubeReal.name:
        env = QQubeReal(args.dt, args.max_steps)
        # policy = QQubeSwingUpAndBalanceCtrl(env.spec)

        # MVD - Learned for the paper
        policy = QQubeSwingUpAndBalanceCtrl(
            env.spec,
            ref_energy=np.exp(-2.9414043),
            energy_gain=np.exp(3.1400251),
            energy_th_gain=0.73774934,  # for simulation and real system
            acc_max=5.,  # Quanser's value: 6
            alpha_max_pd_enable=10.,  # Quanser's value: 20
            pd_gains=to.tensor([-1.9773294, 35.084324, -1.1951622, 3.3797605]))

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
