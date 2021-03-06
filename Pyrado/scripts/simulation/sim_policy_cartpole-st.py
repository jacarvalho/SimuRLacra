"""
Simulate (with animation) a rollout in an environment.
"""
import pprint

import pyrado
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim
from pyrado.environments.pysim.quanser_qube import QQubeSim, QQubeStabSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.base import Policy
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.data_types import RenderMode, EnvSpec
import torch as to
import numpy as np


class CartpoleStabilizerPolicy(Policy):
    """ Swing-up and balancing controller for the Quanser Cart-Pole """

    def __init__(self,
                 env_spec: EnvSpec,
                 K: np.ndarray = np.array([ 1.2278416e+00, 4.5279346e+00 , -1.2385756e-02,  6.0038762e+00,   -4.1818547e+00]),
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

        self._max_u = 3.

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



if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Load the environment and the policy
    env = QCartPoleStabSim(args.dt, args.max_steps)  # runs infinitely by default

    # MVD
    policy = CartpoleStabilizerPolicy(
        env.spec,
        K = np.array([ 1.2278416e+00, 4.5279346e+00 , -1.2385756e-02,  6.0038762e+00,   -4.1818547e+00])
    )

    print_cbt('Set up controller for the QuanserCartpole environment.', 'c')

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
