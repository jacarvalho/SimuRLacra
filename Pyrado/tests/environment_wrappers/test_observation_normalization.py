import numpy as np
import pytest

from pyrado.spaces.box import BoxSpace
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from tests.environment_wrappers.mock_env import MockEnv


@pytest.fixture(scope='function',
                ids=['mock_obs_space'])
def mock_obs_space():
    return BoxSpace([-2, -1, 0], [2, 3, 1])


@pytest.mark.wrappers
def test_space(mock_obs_space):
    mockenv = MockEnv(obs_space=mock_obs_space)
    wenv = ObsNormWrapper(mockenv)
    
    # Check observation space bounds
    lb, ub = wenv.obs_space.bounds
    assert np.all(lb == -1)
    assert np.all(ub == 1)


@pytest.mark.wrappers
def test_denormalization(mock_obs_space):
    mockenv = MockEnv(obs_space=mock_obs_space)
    wenv = ObsNormWrapper(mockenv)

    for _ in range(100):
        # Generate random observations
        obs, _, _, _ = wenv.step(np.array([0, 0, 0]))
        assert (abs(obs) <= 1).all
