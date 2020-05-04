import pytest
from pytest_lazyfixture import lazy_fixture

from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.exploration.stochastic_params import NormalParamNoise
from pyrado.policies.features import *
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat


@pytest.mark.parametrize(
    'env', [
        BallOnBeamSim(dt=0.02, max_steps=1),
        QBallBalancerSim(dt=0.02, max_steps=1),
    ], ids=['bob', 'qbb']
)
@pytest.mark.parametrize(
    'policy', lazy_fixture(
        ['linear_policy', 'fnn_policy']
    ), ids=['linear_policy', 'fnn_policy']
)
def test_noise_on_act(env, policy):
    for _ in range(100):
        # Init the exploration strategy
        act_noise_strat = NormalActNoiseExplStrat(
            policy,
            std_init=0.5,
            train_mean=True
        )

        # Set new parameters for the exploration noise
        std = to.ones(env.act_space.flat_dim)*to.rand(1)
        mean = to.rand(env.act_space.shape)
        act_noise_strat.noise.adapt(mean, std)
        assert (mean == act_noise_strat.noise.mean).all()

        # Sample a random observation from the environment
        obs = to.from_numpy(env.obs_space.sample_uniform())

        # Get a clean and a noisy action
        act = policy(obs)  # policy expects Tensors
        act_noisy = act_noise_strat(obs)  # exploration strategy expects ndarrays
        assert isinstance(act, to.Tensor)
        assert not to.equal(act, act_noisy)


@pytest.mark.parametrize(
    'env', [
        BallOnBeamSim(dt=0.02, max_steps=1),
        QBallBalancerSim(dt=0.02, max_steps=1),
    ], ids=['bob', 'qbb']
)
@pytest.mark.parametrize(
    'policy', lazy_fixture(
        ['linear_policy', 'fnn_policy']
    ), ids=['linear_policy', 'fnn_policy']
)
def test_noise_on_param(env, policy):
    for _ in range(5):
        # Init the exploration strategy
        param_noise_strat = NormalParamNoise(
            policy.num_param,
            full_cov=True,
            std_init=1.,
            std_min=0.01,
            train_mean=True
        )

        # Set new parameters for the exploration noise
        mean = to.rand(policy.num_param)
        cov = to.eye(policy.num_param)
        param_noise_strat.adapt(mean, cov)
        to.testing.assert_allclose(mean, param_noise_strat.noise.mean)

        # Reset exploration strategy
        param_noise_strat.reset_expl_params()

        # Sample a random observation from the environment
        obs = to.from_numpy(env.obs_space.sample_uniform())

        # Get a clean and a noisy action
        act = policy(obs)  # policy expects Tensors
        sampled_param = param_noise_strat.sample_param_set(policy.param_values)
        new_policy = deepcopy(policy)
        new_policy.param_values = sampled_param
        act_noisy = new_policy(obs)  # exploration strategy expects ndarrays

        assert isinstance(act, to.Tensor)
        assert not to.equal(act, act_noisy)
