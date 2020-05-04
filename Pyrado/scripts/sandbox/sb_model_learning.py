"""
Test model learning using PyTorch and the One Mass Oscillator setup.
"""
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim, OneMassOscillatorDomainParamEstimator
from pyrado.policies.dummy import IdlePolicy, DummyPolicy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Set up environment
    dp_gt = dict(m=2., k=20., d=0.8)  # ground truth
    dp_init = dict(m=1.0, k=22., d=0.4)  # initial guess
    dt = 1/50.
    env = OneMassOscillatorSim(dt=dt, max_steps=400)
    env.reset(domain_param=dp_gt)

    # Set up policy
    # policy = IdlePolicy(env.spec)
    policy = DummyPolicy(env.spec)

    # Sample
    sampler = ParallelSampler(env, policy, num_envs=4, min_rollouts=50, seed=1)
    ros = sampler.sample()

    # Create a model for learning the domain parameters
    model = OneMassOscillatorDomainParamEstimator(dt=dt, dp_init=dp_init, num_epoch=50, batch_size=10)

    model.update(ros)

    print_cbt(f'true domain param   : {dp_gt}', 'g')
    print_cbt(f'initial domain param: {dp_init}', 'y')
    print_cbt(f'learned domain param: {model.dp_est.detach().numpy()}', 'c')
