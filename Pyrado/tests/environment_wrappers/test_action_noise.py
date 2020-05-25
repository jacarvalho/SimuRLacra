import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

import pyrado
from pyrado.environment_wrappers.action_noise import GaussianActNoiseWrapper
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from tests.conftest import m_needs_bullet, m_needs_vortex


@pytest.mark.wrappers
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop2d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop2d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop5d_vx'), marks=m_needs_vortex),
    ], ids=['bob', 'qbb', 'bop2d_b', 'bop2d_v', 'bop5d_b', 'bop5d_v']
)
def test_act_noise_simple(env):
    # Typical case with zero mean and non-zero std
    wrapped_env = GaussianActNoiseWrapper(env, noise_std=0.2*np.ones(env.act_space.shape))
    for _ in range(3):
        # Sample some values
        rand_act = env.act_space.sample_uniform()
        wrapped_env.reset()
        obs_nom, _, _, _ = env.step(rand_act)
        obs_wrapped, _, _, _ = wrapped_env.step(rand_act)
        # Different actions can not lead to the same observation
        assert not np.all(obs_nom == obs_wrapped)

    # Unusual case with non-zero mean and zero std
    wrapped_env = GaussianActNoiseWrapper(env, noise_mean=0.1*np.ones(env.act_space.shape))
    for _ in range(3):
        # Sample some values
        rand_act = env.act_space.sample_uniform()
        wrapped_env.reset()
        obs_nom, _, _, _ = env.step(rand_act)
        obs_wrapped, _, _, _ = wrapped_env.step(rand_act)
        # Different actions can not lead to the same observation
        assert not np.all(obs_nom == obs_wrapped)

    # General case with non-zero mean and non-zero std
    wrapped_env = GaussianActNoiseWrapper(env,
                                          noise_mean=0.1*np.ones(env.act_space.shape),
                                          noise_std=0.2*np.ones(env.act_space.shape))
    for _ in range(3):
        # Sample some values
        rand_act = env.act_space.sample_uniform()
        wrapped_env.reset()
        obs_nom, _, _, _ = env.step(rand_act)
        obs_wrapped, _, _, _ = wrapped_env.step(rand_act)
        # Different actions can not lead to the same observation
        assert not np.all(obs_nom == obs_wrapped)


@pytest.mark.wrappers
@pytest.mark.parametrize(
    'env', [
        BallOnBeamSim(dt=0.05, max_steps=1),
    ], ids=['bob']
)
def test_order_act_noise_act_norm(env):
    # First noise wrapper then normalization wrapper
    wrapped_env_noise = GaussianActNoiseWrapper(env,
                                                noise_mean=0.2*np.ones(env.act_space.shape),
                                                noise_std=0.1*np.ones(env.act_space.shape))
    wrapped_env_noise_norm = ActNormWrapper(wrapped_env_noise)

    # First normalization wrapper then noise wrapper
    wrapped_env_norm = ActNormWrapper(env)
    wrapped_env_norm_noise = GaussianActNoiseWrapper(wrapped_env_norm,
                                                     noise_mean=0.2*np.ones(env.act_space.shape),
                                                     noise_std=0.1*np.ones(env.act_space.shape))

    # Sample some values directly from the act_spaces
    for i in range(3):
        pyrado.set_seed(i)
        act_noise_norm = wrapped_env_noise_norm.act_space.sample_uniform()

        pyrado.set_seed(i)
        act_norm_noise = wrapped_env_norm_noise.act_space.sample_uniform()

        # These samples must be the same since were not passed to _process_act function
        assert np.all(act_noise_norm == act_norm_noise)

    # Process a sampled action
    for i in range(3):
        # Sample a small random action such that the denormalization doe not map it to the act_space limits
        rand_act = 0.01*env.act_space.sample_uniform()

        pyrado.set_seed(i)
        o1 = wrapped_env_noise_norm.reset()
        obs_noise_norm, _, _, _ = wrapped_env_noise_norm.step(rand_act)

        pyrado.set_seed(i)
        o2 = wrapped_env_norm_noise.reset()
        obs_norm_noise, _, _, _ = wrapped_env_norm_noise.step(rand_act)

        # The order of processing (first normalization or first randomization must make a difference)
        assert not np.all(obs_noise_norm == obs_norm_noise)
