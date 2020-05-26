import pytest

from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, DomainRandWrapperBuffer


@pytest.mark.wrapper
def test_dr_wrapper_live_bob(default_bob, bob_pert):
    param_init = default_bob.domain_param
    wrapper = DomainRandWrapperLive(default_bob, bob_pert)
    # So far no randomization happened, thus the parameter should be equal
    assert default_bob.domain_param == param_init

    # Randomize 10 times 1 new parameter set
    for _ in range(10):
        param_old = wrapper.domain_param
        wrapper.reset()
        assert param_old != wrapper.domain_param


@pytest.mark.wrapper
def test_dr_wrapper_buffer_bob(default_bob, bob_pert):
    param_init = default_bob.domain_param
    wrapper = DomainRandWrapperBuffer(default_bob, bob_pert)
    # So far no randomization happened, thus the parameter should be equal
    assert default_bob.domain_param == param_init
    assert wrapper._buffer is None

    # Randomize 10 times 13 new parameter sets
    for _ in range(10):
        wrapper.fill_buffer(13)
        for i in range(13):
            param_old = wrapper.domain_param
            assert wrapper._ring_idx == i
            wrapper.reset()
            assert param_old != wrapper.domain_param
