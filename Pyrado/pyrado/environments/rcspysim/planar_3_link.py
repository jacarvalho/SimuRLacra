import functools
import numpy as np
import os.path as osp
from init_args_serializer import Serializable
from typing import Sequence

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.base import Task
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.predefined import create_check_all_boundaries_task
from pyrado.tasks.utils import proximity_succeeded
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, ZeroPerStepRewFcn
from pyrado.tasks.sequential import SequentialTasks
from pyrado.tasks.parallel import ParallelTasks
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'Planar3Link'))


class Planar3LinkSim(RcsSim, Serializable):
    """ Base class for the Planar 3-link environments simulated in Rcs using the Vortex or Bullet physics engine """

    def __init__(self, task_args: dict, max_dist_force: float = None, position_mps: bool = None, **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param max_dist_force: maximum disturbance force, set to None (default) for no disturbance
        :param position_mps: `True` if the MPs are defined on position level, `False` if defined on velocity level,
                             only matters if `actionModelType='activation'`
        :param kwargs: keyword arguments forwarded to `RcsSim`
        """
        Serializable._init(self, locals())

        if kwargs.get('collisionConfig', None) is None:
            collision_config = {
                'pairs': [
                    {'body1': 'Effector', 'body2': 'Link2'},
                    {'body1': 'Effector', 'body2': 'Link1'},
                ],
                'threshold': 0.15,
                'predCollHorizon': 20
            }
        else:
            collision_config = kwargs.get('collisionConfig')

        # Forward to the RcsSim's constructor, nothing more needs to be done here
        RcsSim.__init__(
            self,
            envType='Planar3Link',
            graphFileName='gPlanar3Link.xml',
            task_args=task_args,
            positionTasks=position_mps,
            collisionConfig=collision_config,
            **kwargs
        )

        # Initial state space definition
        upright_init_state = np.array([0.1, 0.1, 0.1])  # [rad, rad, rad]
        self._init_space = SingularStateSpace(upright_init_state, labels=['$q_1$', '$q_2$', '$q_3$'])

        # Setup disturbance
        self._max_dist_force = max_dist_force

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

    def _create_task(self, task_args: dict) -> Task:
        # Define the indices for selection. This needs to match the observations' names in RcsPySim.
        idcs = ['Effector_X', 'Effector_Z']

        # Get the masked environment specification
        spec = EnvSpec(
            self.spec.obs_space,
            self.spec.act_space,
            self.spec.state_space.subspace(self.spec.state_space.create_mask(idcs))
        )

        # Get and set goal position in world coordinates for all three sub-goals
        p1 = self.get_body_position('Goal1', '', '')
        p2 = self.get_body_position('Goal2', '', '')
        p3 = self.get_body_position('Goal3', '', '')
        state_des1 = np.array([p1[0], p1[2]])
        state_des2 = np.array([p2[0], p2[2]])
        state_des3 = np.array([p3[0], p3[2]])

        success_fcn = functools.partial(proximity_succeeded, thold_dist=7.5e-2, dims=[0, 1])  # min distance = 7cm
        Q = np.diag([1e0, 1e0])
        R = 5e-2*np.eye(self.act_space.flat_dim)

        # Create the tasks
        subtask_11 = FinalRewTask(
            DesStateTask(spec, state_des1, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_21 = FinalRewTask(
            DesStateTask(spec, state_des2, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_1p = ParallelTasks(
            [subtask_11, subtask_21], hold_rew_when_done=True, verbose=False
        )
        subtask_3 = FinalRewTask(
            DesStateTask(spec, state_des3, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_12 = FinalRewTask(
            DesStateTask(spec, state_des1, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_22 = FinalRewTask(
            DesStateTask(spec, state_des2, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_2p = ParallelTasks(
            [subtask_12, subtask_22], hold_rew_when_done=True, verbose=False
        )
        task = FinalRewTask(
            SequentialTasks([subtask_1p, subtask_3, subtask_2p], hold_rew_when_done=True, verbose=True),
            mode=FinalRewMode(always_positive=True), factor=2e3
        )
        masked_task = MaskedTask(self.spec, task, idcs)

        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)

        # Return the masked task and and additional task that ends the episode if the unmasked state is out of bound
        return ParallelTasks([masked_task, task_check_bounds])


class Planar3LinkIKSim(Planar3LinkSim, Serializable):
    """
    Planar 3-link robot environment controlled by setting the joint angles, i.e. the agent has to learn the
    inverse kinematics of the 3-link robot
    """

    name: str = 'p3l-ik'

    def __init__(self, state_des: np.ndarray = None, **kwargs):
        """
        Constructor

        :param state_des: desired state for the task
        :param kwargs: keyword arguments forwarded to `RcsSim`
        """
        Serializable._init(self, locals())

        # Forward to the Planar3LinkSim's constructor, specifying the characteristic action model
        super().__init__(task_args=dict(state_des=state_des), actionModelType='joint_vel', **kwargs)  # former joint_pos


class Planar3LinkTASim(Planar3LinkSim, Serializable):
    """ Planar 3-link robot environment controlled by setting the task activation of a Rcs control task """

    name: str = 'p3l-ta'

    def __init__(self,
                 mps: Sequence[dict] = None,
                 collision_config: dict = None,
                 position_mps: bool = True,
                 **kwargs):
        """
        Constructor

        :param mps: movement primitives holding the dynamical systems and the goal states
        :param collision_config: specification of the Rcs `CollisionModel`
        :param position_mps: if `True` use movement primitives specified on position-level, if `False` velocity-level
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'mean',  # 'sum', 'mean',  'product', or 'softmax'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeForceTorque: bool = True,
                       observeCollisionCost: bool = False,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeGoalDistance: bool = False,
                       observeDynamicalSystemDiscrepancy: bool = False,

        Example:
        mps = [{'function': 'lin',
                'errorDynamics': np.eye(dim_mp_state),
                'goal': np.zeros(dim_mp_state)
               },
               {'function': 'lin',
                'errorDynamics':  np.zeros(dim_mp_state),
                'goal': np.ones(dim_mp_state)
               }]

        Example
        mps = [{'function': 'msd_nlin',
                'attractorStiffness': 100.,
                'mass': 1.,
                'damping': 50.,
                'goal': state_des[dim_mp_state]
               },
               {'function': 'msd',
                'attractorStiffness': 100.,
                'mass': 1.,
                'damping': 50.,
                'goal': np.ones(dim_mp_state)
                }]
        """
        Serializable._init(self, locals())

        # Define the movement primitives
        if mps is None:
            if position_mps:
                mps = [
                    {
                        'function': 'msd_nlin',
                        'attractorStiffness': 30.,
                        'mass': 1.,
                        'damping': 50.,
                        'goal': np.array([-0.8, 0.8]),  # position of the left sphere
                    },
                    {
                        'function': 'msd_nlin',
                        'attractorStiffness': 30.,
                        'mass': 1.,
                        'damping': 50.,
                        'goal': np.array([+0.8, 0.8]),  # position of the lower right sphere
                    },
                    {
                        'function': 'msd_nlin',
                        'attractorStiffness': 30.,
                        'mass': 1.,
                        'damping': 50.,
                        'goal': np.array([-0.25, 1.2]),  # position of the upper right sphere
                    }
                ]
            else:
                dt = kwargs.get('dt', 0.01)  # 100 Hz is the default
                mps = [
                    {
                        'function': 'lin',
                        'errorDynamics': 5.*np.eye(2),
                        'goal': dt*np.array([0.06, 0.06])  # X and Z [m/s]
                    },
                    {
                        'function': 'lin',
                        'errorDynamics': 5.*np.eye(2),
                        'goal': dt*np.array([-0.04, -0.04])  # X and Z [m/s]
                    }
                ]

        # Forward to the Planar3LinkSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict(mps=mps),
            actionModelType='activation',
            tasks=mps,
            position_mps=position_mps,
            **kwargs
        )

        # # State space definition
        # if kwargs.get('observeVelocities', True):
        #     self.state_mask = self.obs_space.create_mask(
        #         'Effector_X', 'Effector_Z', 'Effector_Xd', 'Effector_Zd',
        #     )
        # else:
        #     self.state_mask = self.obs_space.create_mask(
        #         'Effector_X', 'Effector_Z'
        #     )
