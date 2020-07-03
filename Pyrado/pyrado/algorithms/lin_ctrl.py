import joblib
import numpy as np
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.environments.base import Env
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate5DSim
from pyrado.environment_wrappers.utils import inner_env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.linear import LinearPolicy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.sampling.sampler import SamplerBase
from pyrado.tasks.reward_functions import QuadrErrRewFcn
from pyrado.utils.tensor import insert_tensor_col


class LQR(Algorithm):
    """ Linear Quadratic Regulator created using the control module """

    name: str = 'lqr'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 num_sampler_envs: int = 4,
                 logger: StepLogger = None,
                 sampler: SamplerBase = None,
                 ball_z_dim_mismatch: bool = True):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy which this algorithm is creating
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param num_sampler_envs: number of environments for parallel sampling
        :param ball_z_dim_mismatch: only useful for BallOnPlate5DSim,
                                    set to True if the controller does not have the z component (relative position)
                                    of the ball in the state vector, i.e. state is 14-dim instead of 16-dim
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(policy, LinearPolicy):
            raise pyrado.TypeErr(given=policy, expected_type=LinearPolicy)

        # Call Algorithm's constructor
        super().__init__(save_dir, 1, policy, logger)

        # Store the inputs
        self._env = env
        self.ball_z_dim_mismatch = ball_z_dim_mismatch

        # Initialize variables for checking and evaluating
        if sampler is None:
            sampler = ParallelSampler(
                env, self._policy,
                num_envs=num_sampler_envs,
                min_steps=min_steps,
                min_rollouts=min_rollouts
            )
        self.sampler = sampler
        self.eigvals = np.array([pyrado.inf])  # initialize with sth positive

    def step(self, snapshot_mode: str, meta_info: dict = None):

        if isinstance(inner_env(self._env), BallOnPlate5DSim):
            ctrl_gains = to.tensor([
                [0.1401, 0, 0, 0, -0.09819, -0.1359, 0, 0.545, 0, 0, 0, -0.01417, -0.04427, 0],
                [0, 0.1381, 0, 0.2518, 0, 0, -0.2142, 0, 0.5371, 0, 0.03336, 0, 0, -0.1262],
                [0, 0, 0.1414, 0.0002534, 0, 0, -0.0002152, 0, 0, 0.5318, 0, 0, 0, -0.0001269],
                [0, -0.479, -0.0004812, 39.24, 0, 0, -15.44, 0, -1.988, -0.001934, 9.466, 0, 0, -13.14],
                [0.3039, 0, 0, 0, 25.13, 15.66, 0, 1.284, 0, 0, 0, 7.609, 6.296, 0]
            ])

            # Compensate for the mismatching different state definition
            if self.ball_z_dim_mismatch:
                ctrl_gains = insert_tensor_col(ctrl_gains, 7, to.zeros((5, 1)))  # ball z position
                ctrl_gains = insert_tensor_col(ctrl_gains, -1, to.zeros((5, 1)))  # ball z velocity

        elif isinstance(inner_env(self._env), QBallBalancerSim):
            # Since the control module can by tricky to install (recommended using anaconda), we only load it if needed
            import control

            # System modeling
            dp = self._env.domain_param
            self._env._calc_constants()
            dp['J_eq'] = self._env._J_eq
            dp['B_eq_v'] = self._env._B_eq_v
            dp['c_kin'] = self._env._c_kin
            dp['zeta'] = self._env._zeta
            dp['A_m'] = self._env._A_m

            A = np.zeros((self._env.obs_space.flat_dim, self._env.obs_space.flat_dim))
            A[:self._env.obs_space.flat_dim//2, self._env.obs_space.flat_dim//2:] = \
                np.eye(self._env.obs_space.flat_dim//2)
            A[4, 4] = -dp['B_eq_v']/dp['J_eq']
            A[5, 5] = -dp['B_eq_v']/dp['J_eq']
            A[6, 0] = dp['c_kin']*dp['m_ball']*dp['g']*dp['r_ball']**2/dp['zeta']
            A[6, 6] = -dp['c_kin']*dp['r_ball']**2/dp['zeta']
            A[7, 1] = dp['c_kin']*dp['m_ball']*dp['g']*dp['r_ball']**2/dp['zeta']
            A[7, 7] = -dp['c_kin']*dp['r_ball']**2/dp['zeta']
            B = np.zeros((self._env.obs_space.flat_dim, self._env.act_space.flat_dim))
            B[4, 0] = dp['A_m']/dp['J_eq']
            B[5, 1] = dp['A_m']/dp['J_eq']
            # C = np.zeros((self._env.obs_space.flat_dim // 2, self._env.obs_space.flat_dim))
            # C[:self._env.obs_space.flat_dim // 2, :self._env.obs_space.flat_dim // 2] =
            # np.eye(self._env.obs_space.flat_dim // 2)
            # D = np.zeros((self._env.obs_space.flat_dim // 2, self._env.act_space.flat_dim))

            # Get the weighting matrices from the environment
            if not isinstance(self._env.task.rew_fcn, QuadrErrRewFcn):
                # The environment uses a reward function compatible with the LQR
                Q = self._env.task.rew_fcn.Q
                R = self._env.task.rew_fcn.R
            else:
                # The environment does not use a reward function compatible with the LQR, apply some fine tuning
                Q = np.diag([1e2, 1e2, 5e2, 5e2, 1e-2, 1e-2, 5e+0, 5e+0])
                R = np.diag([1e-2, 1e-2])

            # Solve the continuous time Riccati eq
            K, _, self.eigvals = control.lqr(A, B, Q, R)  # for discrete system pass dt
            ctrl_gains = to.from_numpy(K).to(to.get_default_dtype())

        else:
            raise pyrado.TypeErr(given=inner_env(self._env), expected_type=[BallOnPlate5DSim, QBallBalancerSim])

        # Assign the controller gains
        self._policy.init_param(-1*ctrl_gains)  # in classical control it is u = -K*x; here a = psi(s)*s

        # Sample rollouts to evaluate the LQR
        ros = self.sampler.sample()

        # Log return -based metrics
        rets = [ro.undiscounted_return() for ro in ros]
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_std = np.std(rets)
        self.logger.add_value('num rollouts', len(ros))
        self.logger.add_value('avg rollout len', np.mean([ro.length for ro in ros]))
        self.logger.add_value('avg return', ret_avg)
        self.logger.add_value('median return', ret_med)
        self.logger.add_value('std return', ret_std)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

    def stopping_criterion_met(self) -> bool:
        """ Checks if the all eigenvalues of the closed loop system are negative. """
        return (self.eigvals < 0).all()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot()

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            joblib.dump(self._env, osp.join(self._save_dir, 'env.pkl'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            pass
