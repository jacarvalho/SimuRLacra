import pytest
import torch as to
import numpy as np
from copy import deepcopy

from pytest_lazyfixture import lazy_fixture
from pyrado.domain_randomization.domain_parameter import NormalDomainParam, MultivariateNormalDomainParam, \
    BernoulliDomainParam
from pyrado.domain_randomization.utils import param_grid
from tests.conftest import m_needs_bullet, m_needs_mujoco


@pytest.mark.sampling
@pytest.mark.parametrize(
    'dp', [
        NormalDomainParam(name='', mean=10, std=1., clip_lo=9, clip_up=11),
        MultivariateNormalDomainParam(name='', mean=to.ones((2, 1)), cov=to.eye(2), clip_lo=-1, clip_up=1.),
        MultivariateNormalDomainParam(name='', mean=10*to.ones((2,)), cov=2*to.eye(2), clip_up=11),
        BernoulliDomainParam(name='', val_0=2, val_1=5, prob_1=0.8),
        BernoulliDomainParam(name='', val_0=-3, val_1=5, prob_1=0.8, clip_up=4),
    ], ids=['1dim', '2dim_v1', '2dim_v2', 'bern_v1', 'bern_v2']
)
def test_domain_param(dp):
    for num_samples in [1, 5, 25]:
        s = dp.sample(num_samples)
        assert len(s) == num_samples


@pytest.mark.sampling
def test_randomizer_dummy(default_dummy_randomizer):
    default_dummy_randomizer.randomize(10)
    samples = default_dummy_randomizer.get_params()
    for i in range(10):
        # The dummy randomizer should always return the nominal parameter values
        assert samples[i] == samples[0]  # only works if none of the values is an array


@pytest.mark.sampling
def test_randomizer(default_pert):
    print(default_pert)
    # Generate 7 samples
    default_pert.randomize(7)

    # Test all variations of the getter function's parameters format and dtype
    pp_3_to_dict = default_pert.get_params(3, format='dict', dtype='torch')
    assert isinstance(pp_3_to_dict, dict)
    assert isinstance(pp_3_to_dict['mass'], list)
    assert len(pp_3_to_dict['mass']) == 3
    assert isinstance(pp_3_to_dict['mass'][0], to.Tensor)
    assert isinstance(pp_3_to_dict['multidim'][0], to.Tensor) and pp_3_to_dict['multidim'][0].shape[0] == 2
    pp_3_to_list = default_pert.get_params(3, format='list', dtype='torch')
    assert isinstance(pp_3_to_list, list)
    assert len(pp_3_to_list) == 3
    assert isinstance(pp_3_to_list[0], dict)
    assert isinstance(pp_3_to_list[0]['mass'], to.Tensor)
    assert isinstance(pp_3_to_list[0]['multidim'], to.Tensor) and pp_3_to_list[0]['multidim'].shape[0] == 2
    pp_3_np_dict = default_pert.get_params(3, format='dict', dtype='numpy')
    assert isinstance(pp_3_np_dict, dict)
    assert isinstance(pp_3_np_dict['mass'], list)
    assert len(pp_3_np_dict['mass']) == 3
    assert isinstance(pp_3_np_dict['mass'][0], np.ndarray)
    assert isinstance(pp_3_np_dict['multidim'][0], np.ndarray) and pp_3_np_dict['multidim'][0].size == 2
    pp_3_np_list = default_pert.get_params(3, format='list', dtype='numpy')
    assert isinstance(pp_3_np_list, list)
    assert len(pp_3_np_list) == 3
    assert isinstance(pp_3_np_list[0], dict)
    assert isinstance(pp_3_np_list[0]['mass'], np.ndarray)
    assert isinstance(pp_3_np_list[0]['multidim'], np.ndarray) and pp_3_np_list[0]['multidim'].size == 2

    pp_all_to_dict = default_pert.get_params(-1, format='dict', dtype='torch')
    assert isinstance(pp_all_to_dict, dict)
    assert isinstance(pp_all_to_dict['mass'], list)
    assert len(pp_all_to_dict['mass']) == 7
    assert isinstance(pp_all_to_dict['mass'][0], to.Tensor)
    assert isinstance(pp_all_to_dict['multidim'][0], to.Tensor) and pp_all_to_dict['multidim'][0].shape[0] == 2
    pp_all_to_list = default_pert.get_params(-1, format='list', dtype='torch')
    assert isinstance(pp_all_to_list, list)
    assert len(pp_all_to_list) == 7
    assert isinstance(pp_all_to_list[0], dict)
    assert isinstance(pp_all_to_list[0]['mass'], to.Tensor)
    assert isinstance(pp_all_to_list[0]['multidim'], to.Tensor) and pp_all_to_list[0]['multidim'].shape[0] == 2
    pp_all_np_dict = default_pert.get_params(-1, format='dict', dtype='numpy')
    assert isinstance(pp_all_np_dict, dict)
    assert isinstance(pp_all_np_dict['mass'], list)
    assert len(pp_all_np_dict['mass']) == 7
    assert isinstance(pp_all_np_dict['mass'][0], np.ndarray)
    assert isinstance(pp_all_np_dict['multidim'][0], np.ndarray) and pp_all_np_dict['multidim'][0].size == 2
    pp_all_np_list = default_pert.get_params(-1, format='list', dtype='numpy')
    assert isinstance(pp_all_np_list, list)
    assert len(pp_all_to_list) == 7
    assert isinstance(pp_all_np_list[0], dict)
    assert isinstance(pp_all_np_list[0]['mass'], np.ndarray)
    assert isinstance(pp_all_np_list[0]['multidim'], np.ndarray) and pp_all_np_list[0]['multidim'].size == 2


def test_rescaling(default_pert):
    # This test relies on a fixed structure of the default_pert ('mass' is ele 0, and 'length is ele 2 in the list).
    randomizer = deepcopy(default_pert)
    randomizer.rescale_distr_param('std', 12.5)
    # Check if the right parameter of the distribution changed
    assert randomizer.domain_params[0].std == 12.5*default_pert.domain_params[0].std
    assert randomizer.domain_params[2].std == 12.5*default_pert.domain_params[2].std
    # Check if the other parameters were unchanged (lazily just check one attribute)
    assert randomizer.domain_params[0].mean == default_pert.domain_params[0].mean
    assert randomizer.domain_params[2].mean == default_pert.domain_params[2].mean


def test_param_grid():
    # Create a parameter grid spec
    pspec = {
        'p1': np.array([0.1, 0.2]),
        'p2': np.array([0.4, 0.5]),
        'p3': 3  # fixed value
    }

    # Create the grid entries
    pgrid = param_grid(pspec)

    # Check for presence of all entries, their order is not mandatory
    assert {'p1': 0.1, 'p2': 0.4, 'p3': 3} in pgrid
    assert {'p1': 0.2, 'p2': 0.4, 'p3': 3} in pgrid
    assert {'p1': 0.1, 'p2': 0.5, 'p3': 3} in pgrid
    assert {'p1': 0.2, 'p2': 0.5, 'p3': 3} in pgrid


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_omo'),
        lazy_fixture('default_pend'),
        lazy_fixture('default_qbb'),
        lazy_fixture('default_qcpst'),
        lazy_fixture('default_qcpsu'),
        pytest.param(lazy_fixture('default_bop2d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_bop5d_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_blpos_bt'), marks=m_needs_bullet),
        pytest.param(lazy_fixture('default_cth'), marks=m_needs_mujoco),
        pytest.param(lazy_fixture('default_hop'), marks=m_needs_mujoco),
        pytest.param(lazy_fixture('default_wambic'), marks=m_needs_mujoco),
    ]
    , ids=['bob', 'omo', 'pend', 'qbb', 'qcp-st', 'qcp-su', 'bop2d', 'bop5d', 'bl_pos', 'cth', 'hop', 'wam-bic']
)
def test_setting_dp_vals(env):
    # Loop over all possible domain parameters and set them to a random value
    for dp_key in env.supported_domain_param:
        rand_val = np.random.rand()  # [0, 1[
        env.reset(domain_param={dp_key: rand_val})
        if any([s in dp_key for s in ['slip', 'compliance', 'linearvelocitydamping', 'angularvelocitydamping']]):
            # Skip the parameters that are only available in Vortex but not in Bullet
            assert True
        else:
            assert env.domain_param[dp_key] == pytest.approx(rand_val, abs=1e-5)
