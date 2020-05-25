import numpy as np
import pytest

from pyrado.spaces.box import BoxSpace
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from tests.environment_wrappers.mock_env import MockEnv


@pytest.mark.wrappers
def test_no_delay():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = ActDelayWrapper(mockenv, delay=0)
    
    # Reset to initialize buffer
    wenv.reset()
    
    # Perform some actions
    wenv.step(np.array([4, 1]))
    assert mockenv.last_act == [4, 1]
    wenv.step(np.array([7, 5]))
    assert mockenv.last_act == [7, 5]


@pytest.mark.wrappers
def test_act_delay():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = ActDelayWrapper(mockenv, delay=2)
    
    # Reset to initialize buffer
    wenv.reset()
    
    # Perform some actions
    wenv.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([2, 4]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [2, 4]


@pytest.mark.wrappers
def test_reset():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = ActDelayWrapper(mockenv, delay=1)
    
    # Reset to initialize buffer
    wenv.reset()
    
    # Perform some actions
    wenv.step(np.array([0, 4]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([4, 4]))
    assert mockenv.last_act == [0, 4]
    
    # The next action would be [4, 4], but now we reset again
    wenv.reset()
    
    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [1, 2]


@pytest.mark.wrappers
def test_domain_param():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = ActDelayWrapper(mockenv, delay=1)
    
    # Reset to initialize buffer
    wenv.reset()
    
    # Perform some actions
    wenv.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([2, 4]))
    assert mockenv.last_act == [0, 1]
    
    # change the delay and reset
    wenv.domain_param = {'act_delay': 2}
    wenv.reset()
    
    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([8, 9]))
    assert mockenv.last_act == [1, 2]
