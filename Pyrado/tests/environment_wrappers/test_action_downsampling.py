import numpy as np
import pytest

from pyrado.spaces.box import BoxSpace
from pyrado.environment_wrappers.downsampling import DownsamplingWrapper
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from tests.environment_wrappers.mock_env import MockEnv


@pytest.mark.wrappers
def test_no_downsampling():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = DownsamplingWrapper(mockenv, factor=1)

    # Perform some actions
    wenv.step(np.array([4, 1]))
    assert mockenv.last_act == [4, 1]
    wenv.step(np.array([7, 5]))
    assert mockenv.last_act == [7, 5]


@pytest.mark.wrappers
def test_act_downsampling():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = DownsamplingWrapper(mockenv, factor=2)

    # Perform some actions
    wenv.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([2, 4]))  # should be ignored
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [1, 2]
    wenv.step(np.array([2, 3]))  # should be ignored
    assert mockenv.last_act == [1, 2]


@pytest.mark.wrappers
def test_reset():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = DownsamplingWrapper(mockenv, factor=2)

    # Perform some actions
    wenv.step(np.array([0, 4]))
    assert mockenv.last_act == [0, 4]
    wenv.step(np.array([4, 4]))
    assert mockenv.last_act == [0, 4]
    wenv.step(np.array([4, 4]))
    assert mockenv.last_act == [4, 4]

    # The next action would be [4, 4] again, but now we reset
    wenv.reset()
    assert wenv._act_last is None
    assert wenv._cnt == 0

    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [1, 2]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [1, 2]


@pytest.mark.wrappers
def test_domain_param():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = DownsamplingWrapper(mockenv, factor=2)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([2, 4]))
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([4, 4]))
    assert mockenv.last_act == [4, 4]

    # change the downsampling and reset
    wenv.domain_param = {'downsampling': 1}
    wenv.reset()

    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [1, 2]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [2, 3]
    wenv.step(np.array([8, 9]))
    assert mockenv.last_act == [8, 9]


@pytest.mark.wrappers
def test_combination_downsampling_delay():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv_ds_dl = DownsamplingWrapper(mockenv, factor=2)
    wenv_ds_dl = ActDelayWrapper(wenv_ds_dl, delay=3)

    # Reset to initialize buffer
    wenv_ds_dl.reset()

    # The first ones are 0 because the ActDelayWrapper's queue is initialized with 0
    wenv_ds_dl.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 0]
    wenv_ds_dl.step(np.array([0, 2]))
    assert mockenv.last_act == [0, 0]
    wenv_ds_dl.step(np.array([0, 3]))
    assert mockenv.last_act == [0, 0]
    wenv_ds_dl.step(np.array([0, 4]))
    # Intuitively one would think last_act would be [0, 1] here, but this is the effect of the wrappers' combination
    assert mockenv.last_act == [0, 0]
    wenv_ds_dl.step(np.array([0, 5]))
    assert mockenv.last_act == [0, 2]
    wenv_ds_dl.step(np.array([0, 6]))
    assert mockenv.last_act == [0, 2]
    wenv_ds_dl.step(np.array([0, 7]))
    assert mockenv.last_act == [0, 4]
    wenv_ds_dl.step(np.array([0, 8]))
    assert mockenv.last_act == [0, 4]
    wenv_ds_dl.step(np.array([0, 9]))
    assert mockenv.last_act == [0, 6]
    wenv_ds_dl.step(np.array([1, 0]))
    assert mockenv.last_act == [0, 6]


@pytest.mark.wrappers
def test_combination_delay_downsampling():
    """ After delay number of actions, the actions are downsampled by the factor """
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv_dl_ds = ActDelayWrapper(mockenv, delay=3)
    wenv_dl_ds = DownsamplingWrapper(wenv_dl_ds, factor=2)

    # Reset to initialize buffer
    wenv_dl_ds.reset()

    # The first ones are 0 because the ActDelayWrapper's queue is initialized with 0
    wenv_dl_ds.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 0]
    wenv_dl_ds.step(np.array([0, 2]))
    assert mockenv.last_act == [0, 0]
    wenv_dl_ds.step(np.array([0, 3]))
    assert mockenv.last_act == [0, 0]
    # One time step earlier than the other order of wrappers
    wenv_dl_ds.step(np.array([0, 4]))
    assert mockenv.last_act == [0, 1]
    wenv_dl_ds.step(np.array([0, 5]))
    assert mockenv.last_act == [0, 1]
    wenv_dl_ds.step(np.array([0, 6]))
    assert mockenv.last_act == [0, 3]
    wenv_dl_ds.step(np.array([0, 7]))
    assert mockenv.last_act == [0, 3]
    wenv_dl_ds.step(np.array([0, 8]))
    assert mockenv.last_act == [0, 5]
    wenv_dl_ds.step(np.array([0, 9]))
    assert mockenv.last_act == [0, 5]
    wenv_dl_ds.step(np.array([1, 0]))
    assert mockenv.last_act == [0, 7]
    wenv_dl_ds.step(np.array([1, 1]))
    assert mockenv.last_act == [0, 7]
