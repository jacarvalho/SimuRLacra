import pytest
import os.path as osp
import torch as to
from pytest_lazyfixture import lazy_fixture

from pyrado.policies.environment_specific import DualRBFLinearPolicy
from tests.conftest import m_needs_cuda, m_needs_bullet, m_needs_mujoco, m_needs_rcs, m_needs_libtorch
from pyrado.policies.adn import ADNPolicy, pd_cubic
from pyrado.policies.dummy import DummyPolicy, IdlePolicy
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import RNNPolicy, default_unpack_hidden, default_pack_hidden
from pyrado.policies.rnn import LSTMPolicy
from pyrado.policies.rnn import GRUPolicy
from pyrado.policies.linear import LinearPolicy
from pyrado.policies.features import *
from pyrado.policies.two_headed import TwoHeadedGRUPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import RenderMode


@pytest.fixture(scope='function',
                ids=['idlepol_bob_default'])
def idlepol_bobspec(default_bob):
    return IdlePolicy(spec=default_bob.spec)


@pytest.fixture(scope='function',
                ids=['dummypol_bob_default'])
def dummypol_bobspec(default_bob):
    return DummyPolicy(spec=default_bob.spec)


@pytest.fixture(scope='function',
                ids=['linpol_bob_default'])
def linpol_bobspec(default_bob, default_fs):
    return LinearPolicy(spec=default_bob.spec, feats=default_fs)


@pytest.fixture(scope='function',
                ids=['fnnpol_bob_default'])
def fnnpol_bobspec(default_bob):
    return FNNPolicy(spec=default_bob.spec, hidden_sizes=(32, 32), hidden_nonlin=to.tanh)


@pytest.fixture(scope='function',
                ids=['cuda_fnnpol_bob_default'])
def cuda_fnnpol_bobspec(default_bob):
    return FNNPolicy(spec=default_bob.spec, hidden_sizes=(32, 32), hidden_nonlin=to.tanh, use_cuda=True)


@pytest.fixture(scope='function',
                ids=['rnnpol_bob_default'])
def rnnpol_bobspec(default_bob):
    return RNNPolicy(spec=default_bob.spec, hidden_size=4, num_recurrent_layers=3, hidden_nonlin='tanh')


@pytest.fixture(scope='function',
                ids=['cuda_rnnpol_bob_default'])
def cuda_rnnpol_bobspec(default_bob):
    return RNNPolicy(spec=default_bob.spec, hidden_size=4, num_recurrent_layers=3, hidden_nonlin='tanh', use_cuda=True)


@pytest.fixture(scope='function',
                ids=['lstmpol_bob_default'])
def lstmpol_bobspec(default_bob):
    return LSTMPolicy(spec=default_bob.spec, hidden_size=4, num_recurrent_layers=3)


@pytest.fixture(scope='function',
                ids=['grupol_bob_default'])
def grupol_bobspec(default_bob):
    return GRUPolicy(spec=default_bob.spec, hidden_size=4, num_recurrent_layers=3)


@pytest.fixture(scope='function',
                ids=['adnpol_bob_default'])
def adnpol_bobspec(default_bob):
    return ADNPolicy(spec=default_bob.spec, dt=default_bob.dt, activation_nonlin=to.sigmoid,
                     potentials_dyn_fcn=pd_cubic)


@pytest.mark.features
@pytest.mark.parametrize(
    'feat_list', [
        [const_feat],
        [identity_feat],
        [const_feat, identity_feat, abs_feat, sign_feat, squared_feat, sin_feat, cos_feat, sinsin_feat, sincos_feat,
         sig_feat, bell_feat],
    ], ids=['const_only', 'ident_only', 'all_simple_feats']
)
def test_simple_feature_stack(feat_list):
    fs = FeatureStack(feat_list)
    obs = to.randn(1, )
    feats_val = fs(obs)
    assert feats_val is not None


@pytest.mark.features
@pytest.mark.parametrize(
    'obs_dim, idcs', [
        (2, [0, 1]),
        (3, [2, 0]),
        (10, [0, 1, 5, 6])
    ], ids=['2_2', '3_2', '10_4']
)
def test_mul_feat(obs_dim, idcs):
    mf = MultFeat(idcs=idcs)
    fs = FeatureStack([identity_feat, mf])
    obs = to.randn(obs_dim, )
    feats_val = fs(obs)
    assert len(feats_val) == obs_dim + 1


@pytest.mark.features
@pytest.mark.parametrize(
    'obs_dim, num_feat_per_dim', [
        (1, 1), (2, 1), (1, 4), (2, 4), (10, 100)
    ], ids=['1_1', '2_1', '1_4', '2_4', '10_100']
)
def test_rff_feat_serial(obs_dim, num_feat_per_dim):
    rff = RandFourierFeat(inp_dim=obs_dim, num_feat_per_dim=num_feat_per_dim, bandwidth=np.ones(obs_dim, ))
    fs = FeatureStack([rff])
    for _ in range(10):
        obs = to.randn(obs_dim, )
        feats_val = fs(obs)
        assert feats_val.shape == (1, num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize('batch_size', [1, 2, 100], ids=['1', '2', '100'])
@pytest.mark.parametrize(
    'obs_dim, num_feat_per_dim', [
        (1, 1), (2, 1), (1, 4), (2, 4), (10, 100)
    ], ids=['1_1', '2_1', '1_4', '2_4', '10_100']
)
def test_rff_feat_batched(batch_size, obs_dim, num_feat_per_dim):
    rff = RandFourierFeat(inp_dim=obs_dim, num_feat_per_dim=num_feat_per_dim, bandwidth=np.ones(obs_dim, ))
    fs = FeatureStack([rff])
    for _ in range(10):
        obs = to.randn(batch_size, obs_dim)
        feats_val = fs(obs)
        assert feats_val.shape == (batch_size, num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    'obs_dim, num_feat_per_dim, bounds', [
        (1, 4, (to.tensor([-3.]), to.tensor([3.]))),
        (1, 4, (np.array([-3.]), np.array([3.]))),
        (2, 4, (to.tensor([-3., -4.]), to.tensor([3., 4.]))),
        (10, 100, (to.tensor([-3.]*10), to.tensor([3.]*10)))
    ], ids=['1_4_to', '1_4_np', '2_4', '10_100']
)
def test_rbf_serial(obs_dim, num_feat_per_dim, bounds):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=bounds)
    fs = FeatureStack([rbf])
    for _ in range(10):
        obs = to.randn(obs_dim, )  # 1-dim obs vector
        feats_val = fs(obs)
        assert feats_val.shape == (1, obs_dim*num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize('batch_size', [1, 2, 100], ids=['1', '2', '100'])
@pytest.mark.parametrize(
    'obs_dim, num_feat_per_dim, bounds', [
        (1, 4, (to.tensor([-3.]), to.tensor([3.]))),
        (1, 4, (np.array([-3.]), np.array([3.]))),
        (2, 4, (to.tensor([-3., -4.]), to.tensor([3., 4.]))),
        (10, 100, (to.tensor([-3.]*10), to.tensor([3.]*10)))
    ], ids=['1_4_to', '1_4_np', '2_4', '10_100']
)
def test_rbf_feat_batched(batch_size, obs_dim, num_feat_per_dim, bounds):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=bounds)
    fs = FeatureStack([rbf])
    for _ in range(10):
        obs = to.randn(batch_size, obs_dim)  # 2-dim obs array
        feats_val = fs(obs)
        assert feats_val.shape == (batch_size, obs_dim*num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qq'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qq', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'num_feat_per_dim', [4, 100], ids=['4', '100']
)
def test_rff_policy_serial(env, num_feat_per_dim):
    rff = RandFourierFeat(inp_dim=env.obs_space.flat_dim, num_feat_per_dim=num_feat_per_dim,
                          bandwidth=env.obs_space.bound_up)
    policy = LinearPolicy(env.spec, FeatureStack([rff]))
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        act = policy(to.from_numpy(obs))
        assert act.shape == (env.act_space.flat_dim,)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qq'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qq', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'batch_size, num_feat_per_dim', [
        (1, 4), (20, 4), (1, 100), (20, 100)
    ], ids=['1_4', '20_4', '1_100', '20_100']
)
def test_rff_policy_batch(env, batch_size, num_feat_per_dim):
    rff = RandFourierFeat(inp_dim=env.obs_space.flat_dim, num_feat_per_dim=num_feat_per_dim,
                          bandwidth=env.obs_space.bound_up)
    policy = LinearPolicy(env.spec, FeatureStack([rff]))
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).repeat(batch_size, 1)
        act = policy(obs)
        assert act.shape == (batch_size, env.act_space.flat_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qq'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qq', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'num_feat_per_dim', [4, 100], ids=['4', '100']
)
def test_rfb_policy_serial(env, num_feat_per_dim):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=env.obs_space.bounds)
    fs = FeatureStack([rbf])
    policy = LinearPolicy(env.spec, fs)
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        act = policy(to.from_numpy(obs))
        assert act.shape == (env.act_space.flat_dim,)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qq'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qq', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'batch_size, num_feat_per_dim', [
        (1, 4), (20, 4), (1, 100), (20, 100)
    ], ids=['1_4', '20_4', '1_100', '20_100']
)
def test_rfb_policy_batch(env, batch_size, num_feat_per_dim):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=env.obs_space.bounds)
    fs = FeatureStack([rbf])
    policy = LinearPolicy(env.spec, fs)
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).repeat(batch_size, 1)
        act = policy(obs)
        assert act.shape == (batch_size, env.act_space.flat_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        pytest.param(lazy_fixture('default_wambic'), marks=m_needs_mujoco),  # so far, the only use case
    ], ids=['wambic']
)
@pytest.mark.parametrize('dim_mask', [0, 1, 2], ids=['0', '1', '2'])
def test_dual_rbf_policy(env, dim_mask):
    # Hyper-parameters for the RBF features are not important here
    rbf_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.]), np.array([1.])), scale=None)
    policy = DualRBFLinearPolicy(env.spec, rbf_hparam, dim_mask)
    assert policy.num_param == policy.num_active_feat*env.act_space.flat_dim//2

    ro = rollout(env, policy, eval=True)
    assert ro is not None


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb')
    ], ids=['bob', 'qbb']
)  # only for using lazy policy fixture
@pytest.mark.parametrize(
    'policy', [
        lazy_fixture('linear_policy'),
        lazy_fixture('fnn_policy'),
        lazy_fixture('rnn_policy'),
        lazy_fixture('lstm_policy'),
        lazy_fixture('gru_policy'),
        lazy_fixture('adn_policy'),
        lazy_fixture('thfnn_policy'),
        lazy_fixture('thgru_policy'),
    ],
    ids=['linear', 'fnn', 'rnn', 'lstm', 'gru', 'adn', 'thfnn', 'thgru']
)
def test_parameterized_policies_init_param(env, policy):
    some_values = to.ones_like(policy.param_values)
    policy.init_param(some_values)
    to.testing.assert_allclose(policy.param_values, some_values)


@pytest.mark.parametrize('policy', lazy_fixture([
    'idlepol_bobspec',
    'dummypol_bobspec',
    'linpol_bobspec',
    'fnnpol_bobspec',
    'cuda_fnnpol_bobspec'
]), ids=['idle', 'dummy', 'linear', 'fnn', 'cuda_fnn'])
def test_feedforward_policy_one_step(policy):
    obs = policy.env_spec.obs_space.sample_uniform()  # shape = (4,)
    obs = to.from_numpy(obs)
    act = policy(obs)
    assert isinstance(act, to.Tensor)


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb')
    ], ids=['bob', 'qbb']
)  # only for using lazy policy fixture
@pytest.mark.parametrize(
    'policy', [
        lazy_fixture('time_policy'),
        lazy_fixture('tracetime_policy'),
    ],
    ids=['time', 'tracetime']
)
def test_time_policy_one_step(env, policy):
    policy.reset()
    obs = policy.env_spec.obs_space.sample_uniform()
    obs = to.from_numpy(obs)
    act = policy(obs)
    assert isinstance(act, to.Tensor)


@pytest.mark.recurrentpolicy
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb')
    ], ids=['bob', 'qbb']
)  # only for using lazy policy fixture
@pytest.mark.parametrize(
    'policy', [
        lazy_fixture('rnn_policy'),
        lazy_fixture('lstm_policy'),
        lazy_fixture('gru_policy'),
        lazy_fixture('adn_policy'),
        lazy_fixture('thgru_policy'),
    ],
    ids=['rnn', 'lstm', 'gru', 'adn', 'thgru']
)
def test_recurrent_policy_one_step(env, policy):
    hid = policy.init_hidden()
    obs = env.obs_space.sample_uniform()
    obs = to.from_numpy(obs)
    if isinstance(policy, TwoHeadedGRUPolicy):
        act, out2, hid = policy(obs, hid)
        assert isinstance(out2, to.Tensor)
    else:
        act, hid = policy(obs, hid)
    assert isinstance(act, to.Tensor) and isinstance(hid, to.Tensor)


@pytest.mark.parametrize('policy', lazy_fixture([
    'linpol_bobspec',
    'fnnpol_bobspec'
]), ids=['linear', 'fnn'])
@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_any_policy_batching(policy, batch_size):
    obs = np.stack([policy.env_spec.obs_space.sample_uniform() for _ in range(batch_size)])  # shape = (batch_size, 4)
    obs = to.from_numpy(obs)
    act = policy(obs)
    assert act.shape[0] == batch_size


def test_linear_policy_one_step_wo_exploration(linpol_bobspec):
    obs = linpol_bobspec.env_spec.obs_space.sample_uniform()  # shape = (4,)
    obs = to.from_numpy(obs)
    val = linpol_bobspec.eval_feats(obs)
    assert isinstance(val, to.Tensor)
    # __call__ uses eval_feats
    act = linpol_bobspec(obs)
    assert isinstance(act, to.Tensor)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize('rnn_policy', lazy_fixture([
    'rnnpol_bobspec'
]), ids=['bob'])
def test_rnn_policy(default_bob, rnn_policy):
    ro = rollout(default_bob, rnn_policy, render_mode=RenderMode(text=True))
    assert isinstance(ro, StepSequence)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize('lstm_policy', lazy_fixture([
    'lstmpol_bobspec'
]), ids=['bob'])
def test_lstm_policy(default_bob, lstm_policy):
    ro = rollout(default_bob, lstm_policy, render_mode=RenderMode(text=True))
    assert isinstance(ro, StepSequence)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize('gru_policy', lazy_fixture([
    'grupol_bobspec'
]), ids=['bob'])
def test_gru_policy(default_bob, gru_policy):
    ro = rollout(default_bob, gru_policy, render_mode=RenderMode(text=True))
    assert isinstance(ro, StepSequence)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize('policy', lazy_fixture([
    'rnnpol_bobspec',
    'lstmpol_bobspec',
    'grupol_bobspec',
    'adnpol_bobspec',
]), ids=['rnn', 'lstm', 'gru', 'adn'])
def test_recurrent_policy(policy):
    obs = policy.env_spec.obs_space.sample_uniform()  # shape = (4,)
    obs = to.from_numpy(obs)

    assert policy.is_recurrent

    # Do this in evaluation mode to disable dropout&co
    policy.eval()

    # Create initial hidden state
    hidden = policy.init_hidden()
    # Use a random one to ensure we don't just run into the 0-special-case
    hidden.random_()
    assert len(hidden) == policy.hidden_size

    # Test general conformity
    act, hid_new = policy(obs, hidden)
    assert len(hid_new) == policy.hidden_size

    # test reproducibility
    act2, hid_new2 = policy(obs, hidden)
    to.testing.assert_allclose(act, act2)
    to.testing.assert_allclose(hid_new2, hid_new2)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'policy', [
        lazy_fixture('rnn_policy'),
        lazy_fixture('lstm_policy'),
        lazy_fixture('gru_policy'),
        lazy_fixture('adn_policy'),
    ], ids=['rnn', 'lstm', 'gru', 'adn']
)
@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_recurrent_policy_batching(env, policy, batch_size):
    obs = np.stack([policy.env_spec.obs_space.sample_uniform() for _ in range(batch_size)])  # shape = (batch_size, 4)
    obs = to.from_numpy(obs)

    assert policy.is_recurrent

    # Do this in evaluation mode to disable dropout&co
    policy.eval()

    # Create initial hidden state
    hidden = policy.init_hidden(batch_size)
    # Use a random one to ensure we don't just run into the 0-special-case
    hidden.random_()
    assert hidden.shape == (batch_size, policy.hidden_size)

    act, hid_new = policy(obs, hidden)
    assert hid_new.shape == (batch_size, policy.hidden_size)

    if batch_size > 1:
        # Try to use a subset of the batch
        subset = to.arange(batch_size//2)
        act_sub, hid_sub = policy(obs[subset, :], hidden[subset, :])

        to.testing.assert_allclose(act_sub, act[subset, :])
        to.testing.assert_allclose(hid_sub, hid_new[subset, :])


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'bop5D']
)
@pytest.mark.parametrize(
    'policy', [
        lazy_fixture('rnn_policy'),
        lazy_fixture('lstm_policy'),
        lazy_fixture('gru_policy'),
        lazy_fixture('adn_policy'),
    ], ids=['rnn', 'lstm', 'gru', 'adn']
)
def test_recurrent_policy_evaluate(env, policy):
    # Make a rollout
    ro = rollout(env, policy, render_mode=RenderMode(text=True))
    ro.torch(to.get_default_dtype())

    # Evaluate first and second action manually
    o1 = ro[0].observation
    h1 = ro[0].hidden_state
    a1, h2 = policy(o1, h1)
    to.testing.assert_allclose(a1.detach(), ro[0].action)
    to.testing.assert_allclose(h2.detach(), ro[0].next_hidden_state)

    # Run evaluate
    eval_act = policy.evaluate(ro)

    to.testing.assert_allclose(eval_act.detach(), ro.actions)


@pytest.mark.recurrent_policy
def test_hidden_state_packing_batch():
    num_layers = 2
    hidden_size = 2
    batch_size = 2

    unpacked = to.tensor([[[1.0, 2.0],  # l1, b1
                           [5.0, 6.0]],  # l1, b2
                          [[3.0, 4.0],  # l2, b1
                           [7.0, 8.0]]])  # l2, b2
    packed = to.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]])

    # Test unpack
    pu = default_unpack_hidden(packed, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(pu, unpacked)

    # Test pack
    up = default_pack_hidden(unpacked, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(up, packed)


@pytest.mark.recurrent_policy
def test_hidden_state_packing_nobatch():
    num_layers = 2
    hidden_size = 2
    batch_size = None

    unpacked = to.tensor([[[1.0, 2.0]],  # l1
                          [[3.0, 4.0]]])  # l2
    packed = to.tensor([1.0, 2.0, 3.0, 4.0])

    # Test unpack
    pu = default_unpack_hidden(packed, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(pu, unpacked)

    # Test pack
    up = default_pack_hidden(unpacked, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(up, packed)


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'policy', [
        # TimePolicy and Two-headed policies are not supported
        lazy_fixture('linear_policy'),
        lazy_fixture('fnn_policy'),
    ], ids=['lin', 'fnn']
)
def test_trace_feedforward(env, policy):
    # Generate scripted version
    scripted = policy.trace()

    # Compare results
    obs = to.from_numpy(policy.env_spec.obs_space.sample_uniform())

    act_reg = policy(obs)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'policy', [
        lazy_fixture('rnn_policy'),
        lazy_fixture('lstm_policy'),
        lazy_fixture('gru_policy'),
        lazy_fixture('adn_policy'),
    ], ids=['rnn', 'lstm', 'gru', 'adn']
)
def test_trace_recurrent(env, policy):
    # Generate scripted version
    scripted = policy.trace()

    # Compare results, tracing hidden manually
    hidden = policy.init_hidden()

    # Run one step
    obs = to.from_numpy(policy.env_spec.obs_space.sample_uniform())
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)
    # Run second step
    obs = to.from_numpy(policy.env_spec.obs_space.sample_uniform())
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)

    # Test after reset
    hidden = policy.init_hidden()
    scripted.reset()

    obs = to.from_numpy(policy.env_spec.obs_space.sample_uniform())
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)


@to.no_grad()
@m_needs_libtorch
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'policy', [
        # TimePolicy and Two-headed policies are not supported
        lazy_fixture('linear_policy'),
        lazy_fixture('fnn_policy'),
        lazy_fixture('rnn_policy'),
        lazy_fixture('lstm_policy'),
        lazy_fixture('gru_policy'),
        lazy_fixture('adn_policy'),
    ], ids=['lin', 'fnn', 'rnn', 'lstm', 'gru', 'adn']
)
def test_export_cpp(env, policy, tmpdir):
    # Generate scripted version (in double mode for CPP compatibility)
    scripted = policy.double().trace()

    # Export
    export_file = osp.join(tmpdir, 'policy.zip')
    scripted.save(export_file)

    # Import again
    loaded = to.jit.load(export_file)

    # Compare a couple of inputs
    for _ in range(50):
        obs = policy.env_spec.obs_space.sample_uniform()
        act_scripted = scripted(to.from_numpy(obs)).cpu().numpy()
        act_loaded = loaded(to.from_numpy(obs)).cpu().numpy()
        assert act_loaded == pytest.approx(act_scripted)

    # Test after reset
    if hasattr(scripted, 'reset'):
        scripted.reset()
        loaded.reset()

        obs = policy.env_spec.obs_space.sample_uniform()
        act_scripted = scripted(to.from_numpy(obs)).numpy()
        act_loaded = loaded(to.from_numpy(obs)).numpy()
        assert act_loaded == pytest.approx(act_scripted)


@to.no_grad()
@m_needs_rcs
@m_needs_libtorch
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb'),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D']
)
@pytest.mark.parametrize(
    'policy', [
        # TimePolicy and Two-headed policies are not supported
        lazy_fixture('linear_policy'),
        lazy_fixture('fnn_policy'),
        lazy_fixture('rnn_policy'),
        lazy_fixture('lstm_policy'),
        lazy_fixture('gru_policy'),
        lazy_fixture('adn_policy'),
    ], ids=['lin', 'fnn', 'rnn', 'lstm', 'gru', 'adn']
)
def test_export_rcspysim(env, policy, tmpdir):
    from rcsenv import ControlPolicy

    # Generate scripted version (in double mode for CPP compatibility)
    scripted = policy.double().trace()
    print(scripted.graph)

    # Export
    export_file = osp.join(tmpdir, 'policy.pt')
    to.jit.save(scripted, export_file)

    # Import in C
    cpp = ControlPolicy('torch', export_file)

    # Compare a couple of inputs
    for _ in range(50):
        obs = policy.env_spec.obs_space.sample_uniform()
        act_script = scripted(to.from_numpy(obs)).numpy()
        act_cpp = cpp(obs, policy.env_spec.act_space.flat_dim)
        assert act_cpp == pytest.approx(act_script)

    # Test after reset
    if hasattr(scripted, 'reset'):
        scripted.reset()
        cpp.reset()
        obs = policy.env_spec.obs_space.sample_uniform()
        act_script = scripted(to.from_numpy(obs)).numpy()
        act_cpp = cpp(obs, policy.env_spec.act_space.flat_dim)
        assert act_cpp == pytest.approx(act_script)


@m_needs_cuda
@pytest.mark.parametrize('policy', lazy_fixture([
    'cuda_fnnpol_bobspec'
]), ids=['cuda_fnn'])
def test_cuda(policy):
    obs = policy.env_spec.obs_space.sample_uniform()  # shape = (4,)
    obs = to.from_numpy(obs)
    act = policy(obs)
    assert 'cuda' in str(act.device)
    assert isinstance(act, to.Tensor)


@m_needs_cuda
@pytest.mark.parametrize('policy', lazy_fixture([
    'cuda_rnnpol_bobspec'
]), ids=['cuda_rnn'])
def test_cuda_rnn(policy):
    obs = policy.env_spec.obs_space.sample_uniform()  # shape = (4,)
    obs = to.from_numpy(obs)
    act, _ = policy(obs)
    assert 'cuda' in str(act.device)
    assert isinstance(act, to.Tensor)
