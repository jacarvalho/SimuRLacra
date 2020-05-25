import numpy as np
import pytest

from pyrado.spaces.box import BoxSpace
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from tests.environment_wrappers.mock_env import MockEnv


@pytest.mark.wrappers
def test_space():
    # Use mock env
    mockenv = MockEnv(act_space=BoxSpace([-2, -1, 0], [2, 3, 1]))
    
    uut = ActNormWrapper(mockenv)
    
    # Check action space bounds
    lb, ub = uut.act_space.bounds
    assert np.all(lb == -1)
    assert np.all(ub == 1)


@pytest.mark.wrappers
def test_denormalization():
    # Use mock env
    mockenv = MockEnv(act_space=BoxSpace([-2, -1, 0], [2, 3, 1]))
    
    uut = ActNormWrapper(mockenv)
    
    # Pass a bunch of actions
    uut.step(np.array([0, 0, 0]))
    assert mockenv.last_act == [0, 1, 0.5]
    
    uut.step(np.array([1, 1, 1]))
    assert mockenv.last_act == [2, 3, 1]
    
    uut.step(np.array([-1, -1, -1]))
    assert mockenv.last_act == [-2, -1, 0]
