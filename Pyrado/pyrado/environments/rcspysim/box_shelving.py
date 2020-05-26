import functools
import numpy as np
import os.path as osp
from init_args_serializer import Serializable
from typing import Sequence

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, MinusOnePerStepRewFcn, RewFcn, AbsErrRewFcn
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.utils import proximity_succeeded
from pyrado.tasks.predefined import create_goal_dist_task, create_check_all_boundaries_task, \
    create_task_space_discrepancy_task, create_collision_task
from pyrado.utils.input_output import print_cbt
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)


def create_box_upper_shelve_task(env_spec: EnvSpec, continuous_rew_fcn: bool, succ_thold: float):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ['Box_X', 'Box_Y', 'Box_Z', 'Box_A', 'Box_B', 'Box_C']

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    state_des = np.zeros(6)  # zeros since we observe the box position relative to the goal
    if continuous_rew_fcn:
        Q = np.diag([5e0, 5e0, 5e0, 1e-1, 1e-1, 1e-1])
        R = 5e-2*np.eye(spec.act_space.flat_dim)
        rew_fcn = ExpQuadrErrRewFcn(Q, R)
    else:
        rew_fcn = MinusOnePerStepRewFcn
    dst = DesStateTask(spec, state_des, rew_fcn, functools.partial(proximity_succeeded, thold_dist=succ_thold))

    # Return the masked tasks
    return MaskedTask(env_spec, dst, idcs)


class BoxShelvingSim(RcsSim, Serializable):
    """ Base class for 2-armed humanoid robot putting a box into a shelve """

    def __init__(self,
                 task_args: dict,
                 ref_frame: str,
                 position_mps: bool,
                 mps_left: [Sequence[dict], None],
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param ref_frame: reference frame for the MPs, e.g. 'world', 'box', or 'upperGoal'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param position_mps: `True` if the MPs are defined on position level, `False` if defined on velocity level
        :param fixed_init_state: use an init state space with only one state (e.g. for debugging)
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'mean',  # 'sum', 'mean',  'product', or 'softmax'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            envType='BoxShelving',
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'BoxShelving'),
            hudColor='BLACK_RUBBER',
            task_args=task_args,
            refFrame=ref_frame,
            positionTasks=position_mps,
            tasksLeft=mps_left,
            **kwargs
        )

        # Initial state space definition
        if fixed_init_state:
            dafault_init_state = np.array([0., 0., 0., 0.8, 30.*np.pi/180, 90.*np.pi/180])  # [m, m, rad, m, rad, rad]
            self._init_space = SingularStateSpace(dafault_init_state,
                                                  labels=['$x$', '$y$', '$th$', '$z$', '$q_2$', '$q_4$'])
        else:
            min_init_state = np.array([-0.02, -0.02, -3.*np.pi/180., 0.78, 27.*np.pi/180, 77.*np.pi/180])
            max_init_state = np.array([0.02, 0.02, 3.*np.pi/180., 0.82, 33.*np.pi/180, 83.*np.pi/180])
            self._init_space = BoxSpace(min_init_state, max_init_state,  # [m, m, rad, m, rad, rad]
                                        labels=['$x$', '$y$', '$th$', '$z$', '$q_2$', '$q_4$'])

    def _create_task(self, task_args: dict) -> Task:
        # Create the tasks
        continuous_rew_fcn = task_args.get('continuous_rew_fcn', True)
        task_box = create_box_upper_shelve_task(self.spec, continuous_rew_fcn, succ_thold=5e-2)
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)
        task_collision = create_collision_task(self.spec, factor=1.)
        task_ts_discrepancy = create_task_space_discrepancy_task(self.spec,
                                                                 AbsErrRewFcn(q=0.5*np.ones(3),
                                                                              r=np.zeros(self.act_space.shape)))

        return ParallelTasks([
            task_box,
            task_check_bounds,
            task_collision,
            task_ts_discrepancy
        ], hold_rew_when_done=False)

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(box_length=0.32,
                    box_width=0.2,
                    box_height=0.1,
                    box_mass=1.,
                    box_friction_coefficient=0.8)


class BoxShelvingPosMPsSim(BoxShelvingSim, Serializable):
    """ Humanoid robot putting a box into a shelve using one arm and position-level movement primitives """

    name: str = 'bs-pos'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None] = None,
                 continuous_rew_fcn: bool = True,
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', 'box', or 'upperGoal'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param fixed_init_state: use an init state space with only one state (e.g. for debugging)
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'mean',  # 'sum', 'mean',  'product', or 'softmax'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Get the nominal domain parameters for the task specification
        dp_nom = BoxShelvingSim.get_nominal_domain_param()

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        if mps_left is None:
            if not ref_frame == 'upperGoal':
                print_cbt(f'Using tasks specified in the upperGoal frame in the {ref_frame} frame!', 'y', bright=True)
            mps_left = [
                # Left power grasp position
                {
                    'function': 'msd', 'attractorStiffness': 30., 'mass': 1., 'damping': 100.,
                    'goal': np.array([0.65, 0, 0.0]),  # far in front
                },
                {
                    'function': 'msd', 'attractorStiffness': 30., 'mass': 1., 'damping': 100.,
                    'goal': np.array([0.35, 0, -0.15]),  # below and in front
                },
                {
                    'function': 'msd', 'attractorStiffness': 30., 'mass': 1., 'damping': 100.,
                    'goal': np.array([0.2, 0, 0.1]),  # close and slightly above
                },
                # Left power grasp orientation
                {
                    'function': 'msd', 'attractorStiffness': 30., 'mass': 1., 'damping': 100.,
                    'goal': np.pi/180*np.array([-90, 0, -90.]),  # upright
                },
                {
                    'function': 'msd', 'attractorStiffness': 30., 'mass': 1., 'damping': 100.,
                    'goal': np.pi/180*np.array([-90, 0, -160.]),  # tilted forward (into shelve)
                },
                # Joints SDH
                {
                    'function': 'msd_nlin', 'attractorStiffness': 50., 'mass': 1., 'damping': 50.,
                    'goal': 10/180*np.pi*np.array([0, 2, -1.5, 2, 0, 2, 0])
                },
            ]

        # Forward to the BoxShelvingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn, mps_left=mps_left),
            mps_left=mps_left,
            ref_frame=ref_frame,
            position_mps=True,
            **kwargs
        )


class BoxShelvingVelMPsSim(BoxShelvingSim, Serializable):
    """ Humanoid robot putting a box into a shelve using one arm and velocity-level movement primitives """

    name: str = 'bs-vel'

    def __init__(self,
                 ref_frame: str,
                 bidirectional_mps: bool,
                 mps_left: [Sequence[dict], None] = None,
                 continuous_rew_fcn: bool = True,
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', 'box', or 'upperGoal'
        :param bidirectional_mps: if `True` then the MPs can be activated "forward" and "backward", thus the `ADN`
                                  output activations must be in [-1, 1] and the output nonlinearity should be a tanh.
                                  If `false` then the behavior is the same as for position-level MPs.
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param fixed_init_state: use an init state space with only one state (e.g. for debugging)
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'mean',  # 'sum', 'mean',  'product', or 'softmax'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Get the nominal domain parameters for the task specification
        dp_nom = BoxShelvingSim.get_nominal_domain_param()

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        if mps_left is None:
            dt = kwargs.get('dt', 0.01)  # 100 Hz is the default

            if bidirectional_mps:
                mps_left = [
                    # Xd
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                    # Yd
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                    # Zd
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                    # Ad
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([10/180*np.pi])},  # [rad/s]
                    # Bd
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([10/180*np.pi])},  # [rad/s]
                    # Joints SDH
                    {
                        'function': 'msd_nlin', 'attractorStiffness': 50., 'mass': 1., 'damping': 50.,
                        'goal': 10/180*np.pi*np.array([0, 2, -1.5, 2, 0, 2, 0])
                    },
                ]
            else:
                mps_left = [
                    # Xd
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.15])},  # [m/s]
                    # Yd
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.15])},  # [m/s]
                    # Zd
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.15])},  # [m/s]
                    # Ad
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([10/180*np.pi])},  # [rad/s]
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-10/180*np.pi])},  # [rad/s]
                    # Bd
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([10/180*np.pi])},  # [rad/s]
                    {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-10/180*np.pi])},  # [rad/s]
                    # Joints SDH
                    {
                        'function': 'msd_nlin', 'attractorStiffness': 50., 'mass': 1., 'damping': 50.,
                        'goal': 10/180*np.pi*np.array([0, 2, -1.5, 2, 0, 2, 0])
                    },
                ]

        # Forward to the BoxShelvingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            mps_left=mps_left,
            ref_frame=ref_frame,
            position_mps=False,
            bidirectionalMPs=bidirectional_mps,
            **kwargs
        )
