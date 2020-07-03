import pytest
import numpy as np
import torch as to
from copy import deepcopy
from pytest_lazyfixture import lazy_fixture
from torch.distributions.normal import Normal

import pyrado
from pyrado.algorithms.adr_discriminator import RewardGenerator
from pyrado.algorithms.utils import compute_action_statistics, until_thold_exceeded
from pyrado.domain_randomization.default_randomizers import get_default_randomizer_omo
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.two_headed import TwoHeadedPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.parallel_sampler import ParallelSampler


@to.no_grad()
@pytest.mark.sampling
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_pend'),
    ]
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
def test_action_statistics(env, policy):
    sigma = 1.  # with lower values like 0.1 we can observe violations of the tolerances

    # Create an action-based exploration strategy
    explstrat = NormalActNoiseExplStrat(policy, std_init=sigma)

    # Sample a deterministic rollout
    pyrado.set_seed(0)
    ro_policy = rollout(env, policy, eval=True, max_steps=1000, stop_on_done=False)
    ro_policy.torch()

    # Run the exploration strategy on the previously sampled rollout
    if policy.is_recurrent:
        if isinstance(policy, TwoHeadedPolicy):
            act_expl, _, _ = explstrat(ro_policy.observations)
        else:
            act_expl, _ = explstrat(ro_policy.observations)
        # Get the hidden states from the deterministic rollout
        hidden_states = ro_policy.hidden_states
    else:
        if isinstance(policy, TwoHeadedPolicy):
            act_expl, _ = explstrat(ro_policy.observations)
        else:
            act_expl = explstrat(ro_policy.observations)
        hidden_states = [0.]*ro_policy.length  # just something that does not violate the format

    ro_expl = StepSequence(
        actions=act_expl[:-1],  # truncate act due to last obs
        observations=ro_policy.observations,
        rewards=ro_policy.rewards,  # don't care but necessary
        hidden_states=hidden_states
    )

    # Compute action statistics and the ground truth
    actstats = compute_action_statistics(ro_expl, explstrat)
    gt_logprobs = Normal(loc=ro_policy.actions, scale=sigma).log_prob(ro_expl.actions)
    gt_entropy = Normal(loc=ro_policy.actions, scale=sigma).entropy()

    to.testing.assert_allclose(actstats.log_probs, gt_logprobs, rtol=1e-4, atol=1e-5)
    to.testing.assert_allclose(actstats.entropy, gt_entropy, rtol=1e-4, atol=1e-5)


@pytest.mark.longtime
@pytest.mark.parametrize(
    'env', [
        lazy_fixture('default_omo'),
    ]
)
def test_adr_reward_generator(env):
    reference_env = env
    random_env = deepcopy(env)
    reward_generator = RewardGenerator(
        env_spec=random_env.spec,
        batch_size=100,
        reward_multiplier=1,
        logger=None
    )
    policy = FNNPolicy(reference_env.spec, hidden_sizes=[32], hidden_nonlin=to.tanh)
    dr = get_default_randomizer_omo()
    dr.randomize(num_samples=1)
    random_env.domain_param = dr.get_params(format='dict', dtype='numpy')
    reference_sampler = ParallelSampler(reference_env, policy, num_envs=4, min_steps=10000)
    random_sampler = ParallelSampler(random_env, policy, num_envs=4, min_steps=10000)

    losses = []
    for i in range(50):
        reference_traj = StepSequence.concat(reference_sampler.sample())
        random_traj = StepSequence.concat(random_sampler.sample())
        losses.append(reward_generator.train(reference_traj, random_traj, 10))
    assert losses[len(losses) - 1] < losses[0]


@pytest.mark.parametrize('thold', [0.5], ids=['0.5'])
@pytest.mark.parametrize('max_iter', [None, 2], ids=['relentless', 'twice'])
def test_until_thold_exceeded(thold, max_iter):
    @until_thold_exceeded(thold, max_iter)
    def _trn_eval_fcn():
        # Draw a random number to mimic a training and evaluation process
        return np.random.rand(1)

    for _ in range(10):
        val = _trn_eval_fcn()
        if max_iter is None:
            assert val >= thold
        else:
            assert True  # there is no easy way to insect the counter, read the printed messages
