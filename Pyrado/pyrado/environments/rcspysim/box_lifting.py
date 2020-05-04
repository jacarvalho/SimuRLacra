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
from pyrado.tasks.endless_flipping import EndlessFlippingTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, MinusOnePerStepRewFcn, AbsErrRewFcn, CosOfOneEleRewFcn, \
    CombinedRewFcn
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.utils import proximity_succeeded, never_succeeded
from pyrado.tasks.predefined import create_check_all_boundaries_task, \
    create_task_space_discrepancy_task, create_collision_task
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)


def create_box_lift_task(env_spec: EnvSpec, continuous_rew_fcn: bool, succ_thold: float):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ['Box_Z']

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    # state_des = np.array([0.3])  # box position is measured relative to the table
    state_des = np.array([1.1])  # box position is measured world coordinates
    if continuous_rew_fcn:
        Q = np.diag([3e1])
        R = 1e0*np.eye(spec.act_space.flat_dim)
        rew_fcn = ExpQuadrErrRewFcn(Q, R)
    else:
        rew_fcn = MinusOnePerStepRewFcn()
    dst = DesStateTask(spec, state_des, rew_fcn, functools.partial(proximity_succeeded, thold_dist=succ_thold))

    # Return the masked tasks
    return MaskedTask(env_spec, dst, idcs)


def create_box_flip_task(env_spec: EnvSpec, continuous_rew_fcn):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ['Box_A']

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    # state_des = np.array([0.3])  # box position is measured relative to the table
    state_des = np.array([-np.pi/2])  # box position is measured world coordinates
    if continuous_rew_fcn:
        q = np.array([0./np.pi])
        r = 1e-6*np.ones(spec.act_space.flat_dim)
        rew_fcn_act = AbsErrRewFcn(q, r)
        rew_fcn_box = CosOfOneEleRewFcn(idx=0)
        rew_fcn = CombinedRewFcn([rew_fcn_act, rew_fcn_box])
    else:
        rew_fcn = MinusOnePerStepRewFcn()
    ef_task = EndlessFlippingTask(spec, rew_fcn, init_angle=0.)

    # Return the masked tasks
    return MaskedTask(env_spec, ef_task, idcs)


class BoxLiftingSim(RcsSim, Serializable):
    """ Base class for 2-armed humanoid robot lifting a box out of a basket """

    def __init__(self,
                 task_args: dict,
                 ref_frame: str,
                 position_mps: bool,
                 mps_left: [Sequence[dict], None],
                 mps_right: [Sequence[dict], None],
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction, e.g `dict(state_des=np.zeros(42))`
        :param ref_frame: reference frame for the position and orientation MPs, e.g. 'world', 'basket', or 'box'
        :param position_mps: `True` if the MPs are defined on position level, `False` if defined on velocity level
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
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
            envType='BoxLifting',
            physicsConfigFile='pBoxLifting.xml',
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'BoxLifting'),
            hudColor='BLACK_RUBBER',
            task_args=task_args,
            refFrame=ref_frame,
            positionTasks=position_mps,
            tasksLeft=mps_left,
            tasksRight=mps_right,
            **kwargs
        )

        # Initial state space definition
        if fixed_init_state:
            dafault_init_state = np.array(
                [0.2, 0., 0., 0.85, 65.*np.pi/180, -65.*np.pi/180])  # [m, m, rad, m, rad, rad]
            self._init_space = SingularStateSpace(dafault_init_state,
                                                  labels=['$x$', '$y$', '$th$', '$z$', '$q_2_L$', '$q_2_R$'])
        else:
            min_init_state = np.array([0.05, -0.05, -5*np.pi/180, 0.8, 60*np.pi/180, -70*np.pi/180])
            max_init_state = np.array([0.25, 0.05, 5*np.pi/180, 0.9, 70*np.pi/180, -60*np.pi/180])
            self._init_space = BoxSpace(min_init_state, max_init_state,  # [m, m, rad, m, rad, rad]
                                        labels=['$x$', '$y$', '$th$', '$z$', '$q_2$', '$q_4$'])

    def _create_task(self, task_args: dict) -> Task:
        # Create the tasks
        continuous_rew_fcn = task_args.get('continuous_rew_fcn', True)
        task_box = create_box_lift_task(self.spec, continuous_rew_fcn, succ_thold=0.03)
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
        return dict(box_length=0.18,
                    box_width=0.14,
                    box_mass=0.3,
                    box_friction_coefficient=1.4,
                    basket_mass=0.5,
                    basket_friction_coefficient=0.6)


class BoxLiftingPosMPsSim(BoxLiftingSim, Serializable):
    """ Humanoid robot lifting a box out of a basket using two arms and position-level movement primitives """

    name: str = 'bl-pos'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None],
                 mps_right: [Sequence[dict], None],
                 continuous_rew_fcn: bool = True,
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the position and orientation MPs, e.g. 'world', 'basket', or 'box'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
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

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        # basket_extends = self.get_body_extents('Basket', 0)
        if mps_left is None:
            mps_left = [
                # Power grasp position in basket frame (basket width = 0.7)
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([0., 0.5, 0.15])},  # [m]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([0., -0.3, 0.15])},  # [m]
                # Power grasp position in box frame (box width = 0.18)
                # {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                #  'goal': np.array([0., 0., 0.1])},  # [m]
                # Power grasp orientation in basket frame
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.pi/180*np.array([180, -90, 0.])},  # [rad]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.pi/180*np.array([120, -90, 0.])},  # [rad]
                # Joints SDH
                {'function': 'msd_nlin', 'attractorStiffness': 50., 'mass': 1., 'damping': 50.,
                 'goal': 10/180*np.pi*np.array([0, 2, -1.5, 2, 0, 2, 0])},
            ]
        if mps_right is None:
            mps_right = [
                # Power grasp position in basket frame (basket width = 0.7)
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([0., -0.5, 0.15])},  # [m]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([0., 0.3, 0.15])},  # [m]
                # Power grasp orientation
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.pi/180*np.array([180, -90, 0.])},  # [rad]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.pi/180*np.array([240, -90, 0.])},  # [rad]
                # Joints SDH
                {'function': 'msd_nlin', 'attractorStiffness': 50., 'mass': 1., 'damping': 50.,
                 'goal': 10/180*np.pi*np.array([0, 1.5, -1, 1, 0, 1.5, 0])},
            ]

        # Forward to the BoxLiftingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            position_mps=True,
            mps_left=mps_left,
            mps_right=mps_right,
            **kwargs
        )


class BoxLiftingVelMPsSim(BoxLiftingSim, Serializable):
    """ Humanoid robot lifting a box out of a basket using two arms and velocity-level movement primitives """

    name: str = 'bl-vel'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None],
                 mps_right: [Sequence[dict], None],
                 continuous_rew_fcn: bool = True,
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the position and orientation MPs, e.g. 'world', 'basket', or 'box'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param fixed_init_state: use an init state space with only one state (e.g. for debugging)
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'mean',  # 'sum', 'mean',  'product', or 'softmax'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeCollisionCost: bool = True,
                       observeVelocities: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        dt = kwargs.get('dt', 0.01)  # 100 Hz is the default
        # basket_extends = self.get_body_extents('Basket', 0)
        if mps_left is None:
            mps_left = [
                # Power grasp Xd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Power grasp Yd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Power grasp Zd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Power grasp Ad
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Power grasp Bd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Power grasp Cd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Joints SDH
                {'function': 'msd_nlin', 'attractorStiffness': 50., 'mass': 2., 'damping': 50.,
                 'goal': 10/180*np.pi*np.array([0, 2, -1.5, 2, 0, 2, 0])},
            ]
        if mps_right is None:
            mps_right = [
                # Power grasp Xd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Power grasp Yd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Power grasp Zd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Power grasp Ad
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Power grasp Bd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Power grasp Cd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Joints SDH
                {'function': 'msd_nlin', 'attractorStiffness': 50., 'mass': 2., 'damping': 50.,
                 'goal': 10/180*np.pi*np.array([0, 1.5, -1, 1, 0, 1.5, 0])},
            ]

        # Forward to the BoxLiftingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            position_mps=False,
            mps_left=mps_left,
            mps_right=mps_right,
            **kwargs
        )


class BoxLiftingSimpleSim(RcsSim, Serializable):
    """ Base class for simplified robotic manipulator turning a box in a basket """

    def __init__(self,
                 task_args: dict,
                 ref_frame: str,
                 position_mps: bool,
                 mps_left: [Sequence[dict], None],
                 **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction, e.g `dict(state_des=np.zeros(42))`
        :param ref_frame: reference frame for the position and orientation MPs, e.g. 'world', 'basket', or 'box'
        :param position_mps: `True` if the MPs are defined on position level, `False` if defined on velocity level
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'mean',  # 'sum', 'mean',  'product', or 'softmax'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = False,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        if kwargs.get('collisionConfig', None) is None:
            kwargs.update(collisionConfig={
                'pairs': [
                    {'body1': 'Hand', 'body2': 'Table'},
                ],
                'threshold': 0.07
            })

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            envType='BoxLiftingSimple',
            physicsConfigFile='pBoxLifting.xml',
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'BoxLifting'),
            hudColor='BLACK_RUBBER',
            task_args=task_args,
            refFrame=ref_frame,
            positionTasks=position_mps,
            tasksLeft=mps_left,
            **kwargs
        )

    def _create_task(self, task_args: dict) -> Task:
        # Create the tasks
        continuous_rew_fcn = task_args.get('continuous_rew_fcn', True)
        task_box = create_box_flip_task(self.spec, continuous_rew_fcn)
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)
        # task_collision = create_collision_task(self.spec, factor=5e-2)
        from pyrado.environments.rcspysim.box_flipping import create_task_space_discrepancy_task
        task_ts_discrepancy = create_task_space_discrepancy_task(self.spec,
                                                                 AbsErrRewFcn(q=5e-2*np.ones(2),
                                                                              r=np.zeros(self.act_space.shape)))

        return ParallelTasks([
            task_box,
            task_check_bounds,
            # task_collision,
            task_ts_discrepancy
        ], hold_rew_when_done=False)

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(box_length=0.18,
                    box_width=0.14,
                    box_mass=0.3,
                    box_friction_coefficient=1.4,
                    basket_mass=0.5,
                    basket_friction_coefficient=0.6)


class BoxLiftingSimplePosMPsSim(BoxLiftingSimpleSim, Serializable):
    """ Simplified robotic manipulator turning a box in a basket using position-level movement primitives """

    name: str = 'bls-pos'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None],
                 continuous_rew_fcn: bool = True,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the position and orientation MPs, e.g. 'world', 'basket', or 'box'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'mean',  # 'sum', 'mean',  'product', or 'softmax'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = False,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        if mps_left is None:
            mps_left = [
                # Y
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([-0.4])},  # [m]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([+0.4])},  # [m]
                # Z
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([-0.05])},  # [m]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([+0.3])},  # [m]
            ]

        # Forward to the BoxLiftingSimpleSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            position_mps=True,
            mps_left=mps_left,
            **kwargs
        )


class BoxLiftingSimpleVelMPsSim(BoxLiftingSimpleSim, Serializable):
    """ Simplified robotic manipulator turning a box in a basket using velocity-level movement primitives """

    name: str = 'bls-vel'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None],
                 continuous_rew_fcn: bool = True,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the position and orientation MPs, e.g. 'world', 'basket', or 'box'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'mean',  # 'sum', 'mean',  'product', or 'softmax'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = False,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        dt = kwargs.get('dt', 0.01)  # 100 Hz is the default
        # basket_extends = self.get_body_extents('Basket', 0)
        if mps_left is None:
            mps_left = [
                # Yd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.1])},  # [m/s]
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.1])},  # [m/s]
                # Zd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.1])},  # [m/s]
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.1])},  # [m/s]
            ]

        # Forward to the BoxLiftingSimpleSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            position_mps=False,
            mps_left=mps_left,
            **kwargs
        )
