import pytest

from pyrado.spaces.box import BoxSpace
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from tests.environment_wrappers.mock_env import MockEnv


@pytest.mark.wrappers
def test_spaces():
    mockenv = MockEnv(obs_space=BoxSpace([-1, -2, -3], [1, 2, 3], labels=['one', 'two', 'three']))

    # Use a simple mask to drop the second element
    mask = [0, 1, 0]
    wenv = ObsPartialWrapper(mockenv, mask)

    # Check resulting space
    lb, ub = wenv.obs_space.bounds
    assert list(lb) == [-1, -3]
    assert list(ub) == [1, 3]
    assert list(wenv.obs_space.labels) == ['one', 'three']


@pytest.mark.wrappers
def test_values():
    mockenv = MockEnv(obs_space=BoxSpace([-1, -2, -3], [1, 2, 3], labels=['one', 'two', 'three']))

    # Use a simple mask to drop the second element
    mask = [0, 1, 0]
    wenv = ObsPartialWrapper(mockenv, mask)

    # Test some observation values
    mockenv.next_obs = [1, 2, 3]
    obs, _, _, _ = wenv.step(None)
    assert list(obs) == [1, 3]

    mockenv.next_obs = [4, 7, 9]
    obs, _, _, _ = wenv.step(None)
    assert list(obs) == [4, 9]


@pytest.mark.wrappers
def test_mask_invert():
    mockenv = MockEnv(obs_space=BoxSpace([-1, -2, -3], [1, 2, 3], labels=['one', 'two', 'three']))

    # Use a simple mask to drop the second element
    mask = [0, 1, 0]
    wenv = ObsPartialWrapper(mockenv, mask, keep_selected=True)

    # Test some observation values
    mockenv.next_obs = [1, 2, 3]
    obs, _, _, _ = wenv.step(None)
    assert list(obs) == [2]

    mockenv.next_obs = [4, 7, 9]
    obs, _, _, _ = wenv.step(None)
    assert list(obs) == [7]


@pytest.mark.wrappers
def test_mask_from_indices():
    # Test the create_mask helper separately
    space = BoxSpace(-1, 1, shape=5)
    indices = [1, 4]

    mask = space.create_mask(indices)
    assert list(mask) == [0, 1, 0, 0, 1]


@pytest.mark.wrappers
def test_mask_from_labels():
    # Test the create_mask helper separately
    space = BoxSpace(-1, 1, shape=5, labels=['w', 'o', 'r', 'l', 'd'])
    indices = ['w', 'o']

    mask = space.create_mask(indices)
    assert list(mask) == [1, 1, 0, 0, 0]
