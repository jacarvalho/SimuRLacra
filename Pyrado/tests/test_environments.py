import pytest
import numpy as np
import torch as to
from pytest_lazyfixture import lazy_fixture

from pyrado.environments.sim_base import SimEnv
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim, QBallBalancerKin
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleStabReal, QCartPoleSwingUpReal
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.utils.data_types import RenderMode
from tests.conftest import m_needs_bullet, m_needs_mujoco, m_needs_vortex


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_cata'),
        lazy_fixture('default_rosen'),
        lazy_fixture('default_bob'),
        lazy_fixture('default_omo'),
        lazy_fixture('default_pend'),
        lazy_fixture('default_qbb'),
        lazy_fixture('default_qq'),
        lazy_fixture('default_qcpst'),
        lazy_fixture('default_qcpsu'),
        pytest.param(lazy_fixture('default_p3l_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_p3l_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop2d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop2d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop5d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bspos_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bspos_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_cth'), marks=m_needs_mujoco),
        pytest.param(lazy_fixture('default_hop'), marks=m_needs_mujoco),
        pytest.param(lazy_fixture('default_wambic'), marks=m_needs_mujoco),
    ], ids=['cata', 'rosen', 'bob', 'omo', 'pend', 'qbb', 'qq', 'qcp-st', 'qcp-su', 'default_p3l_bt', 'default_p3l_vx',
            'bop2d_bt', 'bop2d_vx', 'bop5d_bt', 'bop5d_vx', 'bspos_bt', 'bspos_vx', 'cth', 'hop', 'wam-bic']
)
def test_rollout_dummy(env):
    assert isinstance(env, SimEnv)
    env.reset()
    done = False
    while not done:
        state, rew, done, info = env.step(0.5*env.act_space.sample_uniform())
    assert env.curr_step <= env.max_steps


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_cata'),
        lazy_fixture('default_rosen'),
        lazy_fixture('default_bob'),
        lazy_fixture('default_omo'),
        lazy_fixture('default_pend'),
        lazy_fixture('default_qbb'),
        lazy_fixture('default_qq'),
        lazy_fixture('default_qcpst'),
        lazy_fixture('default_qcpsu'),
        pytest.param(lazy_fixture('default_p3l_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_p3l_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop2d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop2d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop5d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bspos_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bspos_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_cth'), marks=m_needs_mujoco),
        pytest.param(lazy_fixture('default_hop'), marks=m_needs_mujoco),
        pytest.param(lazy_fixture('default_wambic'), marks=m_needs_mujoco),
    ], ids=['cata', 'rosen', 'bob', 'omo', 'pend', 'qbb', 'qq', 'qcp-st', 'qcp-su', 'default_p3l_bt', 'default_p3l_vx',
            'bop2d_bt', 'bop2d_vx', 'bop5d_bt', 'bop5d_vx', 'bspos_bt', 'bspos_vx', 'cth', 'hop', 'wam-bic']
)
def test_init_spaces(env):
    assert isinstance(env, SimEnv)
    # Test using 50 random samples per environment
    for _ in range(50):
        init_space_sample = env.init_space.sample_uniform()
        assert env.init_space.contains(init_space_sample)
        init_obs = env.reset(init_space_sample)
        assert env.obs_space.contains(init_obs)
        assert env.state_space.contains(env.state)


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_cata'),
        lazy_fixture('default_rosen'),
        lazy_fixture('default_bob'),
        lazy_fixture('default_omo'),
        lazy_fixture('default_pend'),
        lazy_fixture('default_qbb'),
        lazy_fixture('default_qq'),
        lazy_fixture('default_qcpst'),
        lazy_fixture('default_qcpsu'),
        pytest.param(lazy_fixture('default_p3l_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_p3l_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop2d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop2d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop5d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bspos_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bspos_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_cth'), marks=m_needs_mujoco),
        pytest.param(lazy_fixture('default_hop'), marks=m_needs_mujoco),
        pytest.param(lazy_fixture('default_wambic'), marks=m_needs_mujoco),
    ], ids=['cata', 'rosen', 'bob', 'omo', 'pend', 'qbb', 'qq', 'qcp-st', 'qcp-su', 'default_p3l_bt', 'default_p3l_vx',
            'bop2d_bt', 'bop2d_vx', 'bop5d_bt', 'bop5d_vx', 'bspos_bt', 'bspos_vx', 'cth', 'hop', 'wam-bic']
)
def test_reset(env):
    assert isinstance(env, SimEnv)
    for _ in range(50):  # do 50 tests
        # Reset the env to a random state
        env.reset()
        env.render(mode=RenderMode(text=True))
        assert env.state_space.contains(env.state, verbose=True)

    # Reset with explicitly specified init state
    initstate = env.init_space.sample_uniform()

    # Explicitly specify once
    obs1 = env.reset(init_state=initstate)
    env.render(mode=RenderMode(text=True))
    assert env.state_space.contains(env.state, verbose=True)

    # Reset to a random state
    env.reset()

    # Reset to fixed state again
    obs2 = env.reset(init_state=initstate)
    # This should match
    assert obs2 == pytest.approx(obs1)


@pytest.mark.visualization
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_omo'),
        lazy_fixture('default_pend'),
        lazy_fixture('default_qbb'),
        lazy_fixture('default_qq'),
        lazy_fixture('default_qcpst'),
    ], ids=['bob', 'omo', 'pend', 'qbb', 'qq', 'qcp-st']
)
def test_vpython_animations(env):
    assert isinstance(env, SimEnv)
    env.reset()
    env.render(mode=RenderMode(video=True))
    for _ in range(300):  # do max 300 steps
        state, rew, done, info = env.step(np.ones(env.act_space.shape))
        env.render(mode=RenderMode(video=True))
        if done:
            break
    assert env.curr_step <= env.max_steps


@pytest.mark.visualization
@pytest.mark.parametrize(
    'env', [
        pytest.param(lazy_fixture('default_p3l_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_p3l_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_pi_6l_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_pi_5l_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop2d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop2d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop5d_vx'), marks=m_needs_vortex),
        pytest.param(lazy_fixture('default_bspos_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bspos_vx'), marks=m_needs_vortex),
    ], ids=['p3l_bt', 'p3l_vx', 'pi_6l_bt', 'pi_5l_vx', 'bop2d_bt', 'bop2d_vx', 'bop5d_bt', 'bop5d_vx',
            'bspos_bt', 'bspos_vx']
)
def test_rcspysim_animations(env):
    assert isinstance(env, SimEnv)
    env.reset()
    env.render(mode=RenderMode(video=True))
    for _ in range(300):  # do max 300 steps
        state, rew, done, info = env.step(np.ones(env.act_space.shape))
        env.render(mode=RenderMode(video=True))
        if done:
            break
    assert env.curr_step <= env.max_steps


@pytest.mark.m_needs_mujoco
@pytest.mark.visualization
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_cth'),
        lazy_fixture('default_hop'),
        lazy_fixture('default_wambic')
    ], ids=['cth', 'hop', 'wam-bic']
)
def test_mujoco_animations(env):
    assert isinstance(env, SimEnv)
    env.reset()
    env.render(mode=RenderMode(video=True))
    for _ in range(300):  # do max 300 steps
        state, rew, done, info = env.step(np.ones(env.act_space.shape))
        env.render(mode=RenderMode(video=True))
        if done:
            break
    assert env.curr_step <= env.max_steps


@pytest.mark.parametrize(
    'servo_ang', [
        np.r_[np.linspace(-np.pi/2.1, np.pi/2.1), np.linspace(np.pi/2.1, -np.pi/2.1)]
    ], ids=['range']
)
def test_qbb_kin(servo_ang):
    env = QBallBalancerSim(dt=0.02, max_steps=100)
    kin = QBallBalancerKin(env, num_opt_iter=50, render_mode=RenderMode(video=False))

    servo_ang = to.tensor(servo_ang, dtype=to.get_default_dtype())
    for th in servo_ang:
        plate_ang = kin(th)
        assert plate_ang is not None


@pytest.mark.parametrize(
    'dt, max_steps', [
        (1/500., 1)
    ], ids=['default']
)
def test_real_env_contructors(dt, max_steps):
    qbbr = QBallBalancerReal(dt=dt, max_steps=max_steps)
    assert qbbr is not None
    qcp_st = QCartPoleStabReal(dt=dt, max_steps=max_steps)
    assert qcp_st is not None
    qcp_su = QCartPoleSwingUpReal(dt=dt, max_steps=max_steps)
    assert qcp_su is not None
    qqr = QQubeReal(dt=dt, max_steps=max_steps)
    assert qqr is not None


@pytest.mark.visualization
@pytest.mark.parametrize(
    'env_name', [
        'MountainCar-v0',
        'CartPole-v1',
        'Acrobot-v1',
        'MountainCarContinuous-v0',
        'Pendulum-v0',
    ], ids=['MountainCar-v0', 'CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0', 'Pendulum-v0']
)
def test_gym_env(env_name):
    # Checking the classic control problems
    gym_module = pytest.importorskip('pyrado.environments.openai_gym')

    env = gym_module.GymEnv(env_name)
    assert env is not None
    env.reset()
    for _ in range(50):
        env.render(RenderMode())
        act = env.act_space.sample_uniform()
        if isinstance(env.act_space, DiscreteSpace):
            act = act.item()
        env.step(act)
    env.close()
