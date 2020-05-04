import functools
import numpy as np
import os.path as osp
from init_args_serializer import Serializable
from typing import Sequence

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.base import Task
from pyrado.tasks.utils import proximity_succeeded
from pyrado.tasks.final_state import FinalRewTask, FinalRewMode
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, MinusOnePerStepRewFcn
from pyrado.tasks.sequential import SequentialTasks
from pyrado.tasks.parallel import ParallelTasks


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'BoxFiddle'))


class PlanarBoxFiddleSim(RcsSim, Serializable):
    """ Base class for the Planar box fiddle environments simulated in Rcs using the Vortex or Bullet physics engine """

    def __init__(self, task_args: dict, max_dist_force: float = None, position_mps: bool = None, **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction, e.g `dict(state_des=np.zeros(42))`
        :param max_dist_force: maximum disturbance force, set to None (default) for no disturbance
        :param position_mps: `True` if the MPs are defined on position level, `False` if defined on velocity level,
                             only matters if `actionModelType='activation'`
        :param kwargs: keyword arguments forwarded to `RcsSim`
        """
        Serializable._init(self, locals())

        # Forward to the RcsSim's constructor, nothing more needs to be done here
        RcsSim.__init__(
            self,
            envType='PlanarBoxFiddle',
            task_args=task_args,
            positionTasks=position_mps,
            **kwargs
        )

        # Store Planar3Link specific vars
        upright_init_state = np.array([0.1, 0.1, 0.1])  # [rad, rad, rad]
        self._init_space = SingularStateSpace(upright_init_state, labels=['$q_1$', '$q_2$', '$q_3$'])

        # Setup disturbance
        self._max_dist_force = max_dist_force

    def _create_task(self, task_args: dict) -> Task:
        # Needs to implemented by subclasses
        raise NotImplementedError

    def _disturbance_generator(self) -> (np.ndarray, None):
        if self._max_dist_force is None:
            return None
        # Sample angle and force uniformly
        angle = np.random.uniform(-np.pi, np.pi)
        force = np.random.uniform(0, self._max_dist_force)
        return np.array([force*np.sin(angle), 0, force*np.cos(angle)])

    @classmethod
    def get_nominal_domain_param(cls):
        return NotImplementedError


class PlanarBoxFiddleIKSim(PlanarBoxFiddleSim, Serializable):
    """
    Planar robot environment controlled by setting the joint angles, i.e. the agent has to learn the
    inverse kinematics of the 3-link robot

    Observation = [eff_x_pos, eff_z_pos, joint1_angpos, joint2_angpos, joint3_angpos, ...
                   eff_x_vel, eff_z_vel, joint1_angvel, joint2_angvel, joint3_angvel]
    Action = [joint1_angpos, joint2_angpos, joint3_angpos]
    """

    name: str = 'pbf'

    def __init__(self, state_des: np.ndarray = None, **kwargs):
        """
        Constructor

        :param state_des: desired state for the task
        :param kwargs: keyword arguments forwarded to `Planar3LinkSim`
        """
        Serializable._init(self, locals())

        # Forward to the Planar3LinkSim's constructor, specifying the characteristic action model
        super().__init__(task_args=dict(state_des=state_des), actionModelType='joint_pos', **kwargs)

        # State space definition
        self.state_mask = self.obs_space.create_mask(
            'Effector_X', 'Effector_Z', 'Effector_Xd', 'Effector_Zd', 'Effector_B', 'Effector_Bd',
        )

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', None)

        if state_des is None:
            state_des = np.array([0, 0, 0, 0, 0, 0])

        success_fcn = functools.partial(proximity_succeeded, thold_dist=5e-2, dims=[0, 1])
        Q = np.diag([1, 1, 1e-1, 1e-1, 1, 1e-1])
        R = 1e-2*np.eye(3)

        # Create the tasks
        return DesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R))
