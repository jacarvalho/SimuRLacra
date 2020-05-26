import pytest
from pytest_lazyfixture import lazy_fixture

from pyrado.algorithms.a2c import A2C
from pyrado.algorithms.adr import ADR
from pyrado.algorithms.hc import HCNormal, HCHyper
from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.reps import REPS
from pyrado.algorithms.sac import SAC
from pyrado.algorithms.nes import NES
from pyrado.algorithms.pepg import PEPG
from pyrado.algorithms.power import PoWER
from pyrado.algorithms.ppo import PPO, PPO2
from pyrado.algorithms.spota import SPOTA
from pyrado.algorithms.svpg import SVPG
from pyrado.domain_randomization.default_randomizers import get_default_randomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.logger import set_log_prefix_dir
from pyrado.policies.features import *
from pyrado.policies.fnn import FNNPolicy, FNN
from pyrado.policies.rnn import RNNPolicy
from pyrado.policies.linear import LinearPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sequences import *
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.data_types import EnvSpec
from tests.conftest import m_needs_bullet


# Fixture providing an experiment directory
@pytest.fixture
def ex_dir(tmpdir):
    set_log_prefix_dir(tmpdir)
    return tmpdir


@pytest.mark.longtime
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_qbb'),  # we just need one env to construct the fixture policies
    ],
    ids=['qbb'],
)
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
    ids=['linear', 'fnn', 'rnn', 'lstm', 'gru', 'adn', 'thfnn', 'thgru'],
)
@pytest.mark.parametrize(
    'algo_class, algo_hparam', [
        (A2C, dict(std_init=0.1)),
        (PPO, dict(std_init=0.1)),
        (PPO2, dict(std_init=0.1)),
        (HCNormal, dict(expl_std_init=0.1, expl_factor=1.1)),
        (HCHyper, dict(expl_r_init=0.05, expl_factor=1.1)),
        (NES, dict(expl_std_init=0.1)),
        (PEPG, dict(expl_std_init=0.1)),
        (PoWER, dict(expl_std_init=0.1, pop_size=5, num_is_samples=5)),
        (REPS, dict(eps=0.1, gamma=0.99, pop_size=100, expl_std_init=0.1)),
    ],
    ids=['a2c', 'ppo', 'ppo2', 'hc_normal', 'hc_hyper', 'nes', 'pepg', 'power', 'reps'])
def test_snapshots_notmeta(ex_dir, env, policy, algo_class, algo_hparam):
    # Collect hyper-parameters, create algorithm, and train
    common_hparam = dict(max_iter=1, num_sampler_envs=1)
    common_hparam.update(algo_hparam)

    if algo_class in [A2C, PPO, PPO2]:
        common_hparam.update(min_rollouts=3,
                             critic=GAE(value_fcn=FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace),
                                                            hidden_sizes=[16, 16],
                                                            hidden_nonlin=to.tanh)))
    elif algo_class in [HCNormal, HCHyper, NES, PEPG, PoWER, REPS]:
        common_hparam.update(num_rollouts=1)
    else:
        raise NotImplementedError

    algo = algo_class(ex_dir, env, policy, **common_hparam)
    algo.train()
    policy_posttrn = deepcopy(algo.policy)  # policy is saved for every algorithm
    if algo_class in [A2C, PPO, PPO2]:
        critic_posttrn = deepcopy(algo.critic)

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo.load_snapshot(load_dir=ex_dir, meta_info=None)
    policy_loaded = deepcopy(algo.policy)
    if algo_class in [A2C, PPO, PPO2]:
        critic_loaded = deepcopy(algo.critic)

    assert all(policy_posttrn.param_values == policy_loaded.param_values)
    if algo_class in [A2C, PPO, PPO2]:
        all(critic_posttrn.value_fcn.param_values == critic_loaded.value_fcn.param_values)


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_bop2d_bt')
    ], ids=['bob', 'bop2d_bt']
)
@pytest.mark.parametrize(
    'algo_class, algo_hparam', [
        (HCNormal, dict(expl_std_init=0.1, expl_factor=1.1)),
        (HCHyper, dict(expl_r_init=0.05, expl_factor=1.1)),
        (NES, dict(expl_std_init=0.1)),
        (NES, dict(expl_std_init=0.1, transform_returns=True)),
        (NES, dict(expl_std_init=0.1, symm_sampling=True)),
        (PEPG, dict(expl_std_init=0.1)),
        (REPS, dict(eps=0.1, gamma=0.99, pop_size=100, expl_std_init=0.1)),
    ],
    ids=['hc_normal', 'hc_hyper', 'nes', 'nes_tr', 'nes_symm', 'pepg', 'reps']
)
def test_param_expl(env, linear_policy, ex_dir, algo_class, algo_hparam):
    # Hyper-parameters
    common_hparam = dict(max_iter=3, num_rollouts=4)
    common_hparam.update(algo_hparam)

    # Create algorithm and train
    algo = algo_class(ex_dir, env, linear_policy, **common_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        pytest.param(lazy_fixture('default_bop2d_bt'), marks=m_needs_bullet),
    ], ids=['bob', 'bop2d_bt']
)
@pytest.mark.parametrize(
    'actor_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=['casual']
)
@pytest.mark.parametrize(
    'value_fcn_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=['casual']
)
@pytest.mark.parametrize(
    'critic_hparam', [dict(gamma=0.995, lamda=1., num_epoch=1, lr=1e-4, standardize_adv=False)], ids=['casual']
)
@pytest.mark.parametrize(
    'algo_hparam', [dict(max_iter=3, num_particles=3, temperature=10, lr=1e-3, horizon=50)], ids=['casual']
)
def test_svpg(env, linear_policy, ex_dir, actor_hparam, value_fcn_hparam, critic_hparam, algo_hparam):
    # Create algorithm and train
    particle_hparam = dict(actor=actor_hparam, value_fcn=value_fcn_hparam, critic=critic_hparam)
    algo = SVPG(ex_dir, env, particle_hparam, **algo_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.metaalgorithm
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_qq')
    ], ids=['qq']
)
@pytest.mark.parametrize(
    'subrtn_hparam', [dict(max_iter=3, min_rollouts=5, num_sampler_envs=1, num_epoch=4)], ids=['casual']
)
@pytest.mark.parametrize(
    'actor_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=['casual']
)
@pytest.mark.parametrize(
    'value_fcn_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=['casual']
)
@pytest.mark.parametrize(
    'critic_hparam', [dict(gamma=0.995, lamda=1., num_epoch=1, lr=1e-4, standardize_adv=False)], ids=['casual']
)
@pytest.mark.parametrize(
    'adr_hparam', [dict(max_iter=3, num_svpg_particles=3, num_discriminator_epoch=3, batch_size=100,
                        num_sampler_envs=1, randomized_params=[])], ids=['casual']
)
def test_adr(env, ex_dir, subrtn_hparam, actor_hparam, value_fcn_hparam, critic_hparam, adr_hparam):
    # Create the subroutine for the meta-algorithm
    actor = FNNPolicy(spec=env.spec, **actor_hparam)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic = GAE(value_fcn, **critic_hparam)
    subroutine = PPO(ex_dir, env, actor, critic, **subrtn_hparam)

    # Create algorithm and train
    particle_hparam = dict(actor=actor_hparam, value_fcn=value_fcn_hparam, critic=critic_hparam)
    algo = ADR(ex_dir, env, subroutine, svpg_particle_hparam=particle_hparam, **adr_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.longtime
@pytest.mark.metaalgorithm
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_qbb')
    ], ids=['qbb']
)
@pytest.mark.parametrize(
    'spota_hparam', [
        dict(max_iter=3, alpha=0.05, beta=0.01, nG=2, nJ=10, ntau=5, nc_init=1, nr_init=1,
             sequence_cand=sequence_add_init, sequence_refs=sequence_const, warmstart_cand=True,
             warmstart_refs=True, num_bs_reps=1000, studentized_ci=False),
    ], ids=['casual_hparam']
)
def test_spota_ppo(env, spota_hparam, ex_dir):
    # Environment and domain randomization
    randomizer = get_default_randomizer(env)
    env = DomainRandWrapperBuffer(env, randomizer)

    # Policy and subroutines
    policy = FNNPolicy(env.spec, [16, 16], hidden_nonlin=to.tanh)
    value_fcn = FNN(input_size=env.obs_space.flat_dim, output_size=1, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    critic_hparam = dict(gamma=0.998, lamda=0.95, num_epoch=3, batch_size=64, lr=1e-3)
    critic_cand = GAE(value_fcn, **critic_hparam)
    critic_refs = GAE(deepcopy(value_fcn), **critic_hparam)

    subrtn_hparam_cand = dict(
        # min_rollouts=0,  # will be overwritten by SPOTA
        min_steps=0,  # will be overwritten by SPOTA
        max_iter=2, num_epoch=3, eps_clip=0.1, batch_size=64, num_sampler_envs=4, std_init=0.5, lr=1e-2)
    subrtn_hparam_cand = subrtn_hparam_cand

    sr_cand = PPO(ex_dir, env, policy, critic_cand, **subrtn_hparam_cand)
    sr_refs = PPO(ex_dir, env, deepcopy(policy), critic_refs, **subrtn_hparam_cand)

    # Create algorithm and train
    algo = SPOTA(ex_dir, env, sr_cand, sr_refs, **spota_hparam)
    algo.train()


@pytest.mark.algorithm
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_bob'),
        lazy_fixture('default_qbb')
    ], ids=['bob', 'qbb']
)
@pytest.mark.parametrize(
    'algo, algo_hparam',
    [
        (A2C, dict()),
        (PPO, dict()),
        (PPO2, dict()),
    ],
    ids=['a2c', 'ppo', 'ppo2'])
@pytest.mark.parametrize(
    'value_fcn_type',
    [
        'fnn-plain',
        'fnn',
        'rnn',
    ],
    ids=['vf_fnn_plain', 'vf_fnn', 'vf_rnn']
)
@pytest.mark.parametrize('use_cuda', [False, True], ids=['cpu', 'cuda'])
def test_actor_critic(env, linear_policy, ex_dir, algo, algo_hparam, value_fcn_type, use_cuda):
    # Create value function
    if value_fcn_type == 'fnn-plain':
        value_fcn = FNN(
            input_size=env.obs_space.flat_dim,
            output_size=1,
            hidden_sizes=[16, 16],
            hidden_nonlin=to.tanh,
            use_cuda=use_cuda
        )
    else:
        vf_spec = EnvSpec(env.obs_space, ValueFunctionSpace)
        if value_fcn_type == 'fnn':
            value_fcn = FNNPolicy(
                vf_spec,
                hidden_sizes=[16, 16],
                hidden_nonlin=to.tanh,
                use_cuda=use_cuda
            )
        else:
            value_fcn = RNNPolicy(
                vf_spec,
                hidden_size=16,
                num_recurrent_layers=1,
                use_cuda=use_cuda
            )

    # Create critic
    critic_hparam = dict(
        gamma=0.98,
        lamda=0.95,
        batch_size=32,
        lr=1e-3,
        standardize_adv=False,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Common hyper-parameters
    common_hparam = dict(max_iter=3, min_rollouts=3, num_sampler_envs=1)
    # Add specific hyper parameters if any
    common_hparam.update(algo_hparam)

    # Create algorithm and train
    algo = algo(ex_dir, env, linear_policy, critic, **common_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.longtime
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_omo')
    ], ids=['omo']
)
@pytest.mark.parametrize(
    'algo_hparam', [
        dict(max_iter=30, pop_size=40, num_rollouts=6, num_is_samples=20, expl_std_init=1.0),
    ], ids=['casual_hparam']
)
def test_power_training(env, algo_hparam, ex_dir):
    # Environment and policy
    policy_hparam = dict(feats=FeatureStack([const_feat, identity_feat]))
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Get init return for comparison
    ret_before = rollout(env, policy, eval=True).undiscounted_return()

    # Create algorithm and train
    algo = PoWER(ex_dir, env, policy, **algo_hparam)
    algo.train()

    # Compare returns befor and after training for max_iter iteration
    ret_after = rollout(env, policy, eval=True).undiscounted_return()
    assert ret_after > ret_before


@pytest.mark.algorithm
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_omo')
    ], ids=['omo']
)
@pytest.mark.parametrize(
    'module', [
        lazy_fixture('linear_policy'),
        lazy_fixture('fnn_policy'),
        lazy_fixture('rnn_policy'),
        lazy_fixture('lstm_policy'),
        lazy_fixture('gru_policy'),
    ]
    , ids=['linear', 'fnn', 'rnn', 'lstm', 'gru']
)
def test_soft_update(env, module):
    # Init param values
    target, source = deepcopy(module), deepcopy(module)
    target.param_values = to.zeros_like(target.param_values)
    source.param_values = to.ones_like(source.param_values)

    # Do one soft update
    SAC.soft_update(target, source, tau=0.8)
    assert to.allclose(target.param_values, 0.2*to.ones_like(target.param_values))

    # Do a second soft update to see the exponential decay
    SAC.soft_update(target, source, tau=0.8)
    assert to.allclose(target.param_values, 0.36*to.ones_like(target.param_values))
