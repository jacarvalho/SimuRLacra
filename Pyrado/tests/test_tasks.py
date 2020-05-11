import functools
import pytest
import numpy as np

from pyrado.utils.data_types import EnvSpec
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.sequential import SequentialTasks
from pyrado.tasks.utils import proximity_succeeded
from pyrado.tasks.desired_state import DesStateTask, RadiallySymmDesStateTask
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.reward_functions import CombinedRewFcn, CosOfOneEleRewFcn, MinusOnePerStepRewFcn, QuadrErrRewFcn, \
    ScaledExpQuadrErrRewFcn, RewFcn


@pytest.mark.parametrize(
    'fcn_list, reset_args, reset_kwargs', [
        ([MinusOnePerStepRewFcn()], [None], [None]),
        ([CosOfOneEleRewFcn(0)], [None], [None]),
        ([QuadrErrRewFcn(np.eye(2), np.eye(1))], [None], [None]),
        ([MinusOnePerStepRewFcn(), QuadrErrRewFcn(Q=np.eye(2), R=np.eye(1))], [None, None], [None, None]),
    ], ids=['wo_args-wo_kwargs', 'w_args-wo_kwargs', 'w_args2-wo_kwargs', 'wo_args-w_kwargs'])
def test_combined_reward_function_step(fcn_list, reset_args, reset_kwargs):
    # Create combined reward function
    c = CombinedRewFcn(fcn_list)
    # Create example state and action error
    err_s, err_a = np.array([1., 2.]), np.array([3.])
    # Calculate combined reward
    rew = c(err_s, err_a)
    assert isinstance(rew, float)
    # Reset the reward functions
    c.reset(reset_args, reset_kwargs)


def test_modulated_rew_fcn():
    Q = np.eye(4)
    R = np.eye(2)
    s = np.array([1, 2, 3, 4])
    a = np.array([0, 0])

    # Modulo 2 for all selected states
    idcs = [0, 1, 3]
    rew_fcn = QuadrErrRewFcn(Q, R)
    task = RadiallySymmDesStateTask(EnvSpec(None, None, None), np.zeros(4), rew_fcn, idcs, 2)
    r = task.step_rew(s, a)
    assert r == -(1 ** 2 + 3 ** 2)

    # Different modulo factor for the selected states
    idcs = [1, 3]
    task = RadiallySymmDesStateTask(EnvSpec(None, None, None), np.zeros(4), rew_fcn, idcs, np.array([2, 3]))
    r = task.step_rew(s, a)
    assert r == -(1 ** 2 + 3 ** 2 + 1 ** 2)


@pytest.mark.parametrize(
    'state_space, act_space', [
        (BoxSpace(-np.ones((7,)), np.ones((7,))), BoxSpace(-np.ones((3,)), np.ones((3,)))),
    ], ids=['box']
)
def test_rew_fcn_constructor(state_space, act_space):
    r_m1 = MinusOnePerStepRewFcn()
    r_quadr = QuadrErrRewFcn(Q=5*np.eye(4), R=2*np.eye(1))
    r_exp = ScaledExpQuadrErrRewFcn(Q=np.eye(7), R=np.eye(3), state_space=state_space, act_space=act_space)
    assert r_m1 is not None
    assert r_quadr is not None
    assert r_exp is not None


@pytest.mark.parametrize(
    'task_type', [
        'ParallelTasks',
        'SequentialTasks'
    ], ids=['parallel', 'sequential']
)
def test_composite_task_structure(task_type):
    env_spec = EnvSpec(obs_space=BoxSpace(-1, 1, 3), act_space=BoxSpace(-1, 1, 2), state_space=BoxSpace(-1, 1, 3))
    state_des1 = np.zeros(3)
    state_des2 = -.5*np.ones(3)
    state_des3 = +.5*np.ones(3)
    rew_fcn = MinusOnePerStepRewFcn()
    t1 = FinalRewTask(DesStateTask(env_spec, state_des1, rew_fcn), mode=FinalRewMode(always_positive=True), factor=10)
    t2 = FinalRewTask(DesStateTask(env_spec, state_des2, rew_fcn), mode=FinalRewMode(always_positive=True), factor=10)
    t3 = FinalRewTask(DesStateTask(env_spec, state_des3, rew_fcn), mode=FinalRewMode(always_positive=True), factor=10)

    if task_type == 'ParallelTasks':
        ct = ParallelTasks([t1, t2, t3])
    elif task_type == 'SequentialTasks':
        ct = SequentialTasks([t1, t2, t3])
    else:
        raise NotImplementedError
    ct.reset(env_spec=env_spec)

    assert len(ct) == 3
    assert ct.env_spec.obs_space == env_spec.obs_space
    assert ct.env_spec.act_space == env_spec.act_space
    assert ct.env_spec.state_space == env_spec.state_space
    assert isinstance(ct.tasks[0].rew_fcn, RewFcn)
    assert isinstance(ct.tasks[1].rew_fcn, RewFcn)
    assert isinstance(ct.tasks[2].rew_fcn, RewFcn)

    if type == 'ParallelTasks':
        assert np.all(ct.state_des[0] == state_des1)
        assert np.all(ct.state_des[1] == state_des2)
        assert np.all(ct.state_des[2] == state_des3)
    elif type == 'SequentialTasks':
        assert np.all(ct.state_des == state_des1)


@pytest.mark.parametrize(
    'hold_rew_when_done', [
        True,
        False
    ], ids=['hold_rews', 'dont_hold_rews']
)
def test_parallel_task_function(hold_rew_when_done):
    # Create env spec and sub-tasks (state_space is necessary for the has_failed function)
    env_spec = EnvSpec(obs_space=BoxSpace(-1, 1, 3), act_space=BoxSpace(-1, 1, 2), state_space=BoxSpace(-1, 1, 3))
    state_des1 = np.zeros(3)
    state_des2 = -.5*np.ones(3)
    state_des3 = +.5*np.ones(3)
    rew_fcn = MinusOnePerStepRewFcn()
    succ_fcn = functools.partial(proximity_succeeded, thold_dist=1e-6)  # necessary to stop a sub-task on success
    t1 = FinalRewTask(DesStateTask(env_spec, state_des1, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)
    t2 = FinalRewTask(DesStateTask(env_spec, state_des2, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)
    t3 = FinalRewTask(DesStateTask(env_spec, state_des3, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)

    pt = FinalRewTask(ParallelTasks([t1, t2, t3], hold_rew_when_done),
                      mode=FinalRewMode(always_positive=True), factor=100)

    # Create artificial dynamics by hard-coding a sequence of states
    num_steps = 12
    fixed_traj = np.linspace(-.5, +.6, num_steps, endpoint=True)  # for the final step, all sub-tasks would be true
    r = [None]*num_steps

    for i in range(num_steps):
        # Advance the artificial state
        state = fixed_traj[i]*np.ones(3)

        # Get all sub-tasks step rew, check if they are done, and if so
        r[i] = pt.step_rew(state, act=np.zeros(2), remaining_steps=11 - i)

        # Check if reaching the sub-goals is recognized
        if np.all(state == state_des1):
            assert pt.wrapped_task.tasks[0].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 7  # only true for this specific setup
            else:
                assert r[i] == 8  # only true for this specific setup
        if np.all(state == state_des2):
            assert pt.wrapped_task.tasks[1].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 7  # only true for this specific setup
            else:
                assert r[i] == 7  # only true for this specific setup
        if np.all(state == state_des3):
            assert pt.wrapped_task.tasks[2].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 7  # only true for this specific setup
            else:
                assert r[i] == 9  # only true for this specific setup

        if i < 10:
            # The combined task is not successful until all sub-tasks are successful
            assert not pt.has_succeeded(state)
        elif i == 10:
            # Should succeed on the second to last
            assert pt.has_succeeded(state)
            assert pt.final_rew(state, 0) == pytest.approx(100.)
        elif i == 11:
            # The very last step reward
            if hold_rew_when_done:
                assert r[i] == -3.
            else:
                assert r[i] == 0.
            assert pt.final_rew(state, 0) == pytest.approx(0.)  # only yield once


@pytest.mark.parametrize(
    'hold_rew_when_done', [
        True,
        False
    ], ids=['hold_rews', 'dont_hold_rews']
)
def test_sequential_task_function(hold_rew_when_done):
    # Create env spec and sub-tasks (state_space is necessary for the has_failed function)
    env_spec = EnvSpec(obs_space=BoxSpace(-1, 1, 3), act_space=BoxSpace(-1, 1, 2), state_space=BoxSpace(-1, 1, 3))
    state_des1 = -.5*np.ones(3)
    state_des2 = np.zeros(3)
    state_des3 = +.5*np.ones(3)
    rew_fcn = MinusOnePerStepRewFcn()
    succ_fcn = functools.partial(proximity_succeeded, thold_dist=1e-6)  # necessary to stop a sub-task on success
    t1 = FinalRewTask(DesStateTask(env_spec, state_des1, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)
    t2 = FinalRewTask(DesStateTask(env_spec, state_des2, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)
    t3 = FinalRewTask(DesStateTask(env_spec, state_des3, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)

    st = FinalRewTask(SequentialTasks([t1, t2, t3], 0, hold_rew_when_done),
                      mode=FinalRewMode(always_positive=True), factor=100)

    # Create artificial dynamics by hard-coding a sequence of states
    num_steps = 12
    fixed_traj = np.linspace(-.5, +.6, num_steps, endpoint=True)  # for the final step, all sub-tasks would be true
    r = [None]*num_steps

    for i in range(num_steps):
        # Advance the artificial state
        state = fixed_traj[i]*np.ones(3)

        # Get all sub-tasks step rew, check if they are done, and if so
        r[i] = st.step_rew(state, act=np.zeros(2), remaining_steps=11 - i)

        # Check if reaching the sub-goals is recognized
        if np.all(state == state_des1):
            assert st.wrapped_task.tasks[0].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 9  # only true for this specific setup
            else:
                assert r[i] == 9  # only true for this specific setup
        if np.all(state == state_des2):
            assert st.wrapped_task.tasks[1].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 8  # only true for this specific setup
            else:
                assert r[i] == 9  # only true for this specific setup
        if np.all(state == state_des3):
            assert st.wrapped_task.tasks[2].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 7  # only true for this specific setup
            else:
                assert r[i] == 9  # only true for this specific setup

        if i < 10:
            # The combined task is not successful until all sub-tasks are successful
            assert not st.has_succeeded(state)
        elif i == 10:
            # Should succeed on the second to last
            assert st.has_succeeded(state)
            assert st.final_rew(state, 0) == pytest.approx(100.)
        elif i == 11:
            # The very last step reward
            if hold_rew_when_done:
                assert r[i] == -3.
            else:
                assert r[i] == 0.
            assert st.final_rew(state, 0) == pytest.approx(0.)  # only yield once
