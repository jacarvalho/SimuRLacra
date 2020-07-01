import functools
import numpy as np

import pyrado
from pyrado.spaces.empty import EmptySpace
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.reward_functions import RewFcn, ZeroPerStepRewFcn, AbsErrRewFcn
from pyrado.tasks.utils import proximity_succeeded, never_succeeded
from pyrado.utils.data_types import EnvSpec


def create_goal_dist_task(env_spec: EnvSpec,
                          ds_index: int,
                          rew_fcn: RewFcn,
                          succ_thold: float = 0.01) -> MaskedTask:
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = [f'GD_DS{ds_index}']

    # Get the masked environment specification
    spec = EnvSpec(
            env_spec.obs_space,
            env_spec.act_space,
            env_spec.state_space.subspace(
                    env_spec.state_space.create_mask(idcs)) if env_spec.state_space is not EmptySpace else EmptySpace
    )

    # Create a desired state task with the goal [0, 0]
    dst = DesStateTask(spec, np.zeros(2), rew_fcn, functools.partial(proximity_succeeded, thold_dist=succ_thold))

    # Mask selected goal distance
    return MaskedTask(env_spec, dst, idcs)


def create_goal_dist_distvel_task(env_spec: EnvSpec,
                                  ds_index: int,
                                  rew_fcn: RewFcn,
                                  succ_thold: float = 0.01) -> MaskedTask:
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = [f'GD_DS{ds_index}', f'GD_DS{ds_index}d']

    # Get the masked environment specification
    spec = EnvSpec(
            env_spec.obs_space,
            env_spec.act_space,
            env_spec.state_space.subspace(
                    env_spec.state_space.create_mask(idcs)) if env_spec.state_space is not EmptySpace else EmptySpace
    )

    # Create a desired state task with the goal [0, 0]
    dst = DesStateTask(spec, np.zeros(2), rew_fcn, functools.partial(proximity_succeeded, thold_dist=succ_thold))

    # Mask selected goal distance velocities
    return MaskedTask(env_spec, dst, idcs)


def create_check_all_boundaries_task(env_spec: EnvSpec, penalty: float) -> FinalRewTask:
    # Check every limit (nut just of a subspace of the state state as it could happen when using a MaskedTask)
    return FinalRewTask(
        DesStateTask(env_spec, np.zeros(env_spec.state_space.shape), ZeroPerStepRewFcn(), never_succeeded),
        FinalRewMode(always_negative=True), factor=penalty
    )


def create_task_space_discrepancy_task(env_spec: EnvSpec, rew_fcn: RewFcn) -> MaskedTask:
    """
    Create a task which punishes the discrepancy between the actual and the commanded state of the observed body.
    The observed body is specified in in the associated experiment configuration file in RcsPySim.
    This task only looks at the X and Z coordinates.

    :param env_spec: environment specification
    :param rew_fcn: reward function
    :return: masked task
    """
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = [idx for idx in env_spec.state_space.labels if 'DiscrepTS' in idx]
    if not idcs:
        raise pyrado.ValueErr(msg="No state space labels found that contain 'DiscrepTS'")

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task (no task space discrepancy is desired and the task never stops because of success)
    dst = DesStateTask(spec, np.zeros(spec.state_space.shape), rew_fcn, never_succeeded)

    # Mask selected discrepancy observation
    return MaskedTask(env_spec, dst, idcs)


def create_collision_task(env_spec: EnvSpec, factor: float) -> MaskedTask:
    """
    Create a task which punishes collision costs given a collision model with pairs of bodies.
    This task only looks at the instantaneous collision cost.

    :param env_spec: environment specification
    :param factor: cost / reward function scaling factor
    :return: masked task
    """
    if not factor >= 0:
        raise pyrado.ValueErr(given=factor, ge_constraint='0')

    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ['CollCost']

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    rew_fcn = AbsErrRewFcn(q=np.array([factor]), r=np.zeros(spec.act_space.shape))

    # Create a desired state task (no collision is desired and the task never stops because of success)
    dst = DesStateTask(spec, np.zeros(spec.state_space.shape), rew_fcn, never_succeeded)

    # Mask selected collision cost observation
    return MaskedTask(env_spec, dst, idcs)
