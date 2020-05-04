"""
Script to sample some rollouts using the ParallelSampler
"""
from tabulate import tabulate

from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.policies.features import FeatureStack, identity_feat, squared_feat
from pyrado.policies.linear import LinearPolicy
from pyrado.sampling.parallel_sampler import ParallelSampler


if __name__ == '__main__':
    # Set up environment
    env = BallOnBeamSim(dt=0.02, max_steps=500)
    env = ActNormWrapper(env)

    # Set up policy
    feats = FeatureStack([identity_feat, squared_feat])
    policy = LinearPolicy(env.spec, feats)

    # Set up sampler
    sampler = ParallelSampler(env, policy, num_envs=2, min_rollouts=2000)

    # Sample and print
    ros = sampler.sample()
    print(tabulate({'StepSequence count': len(ros),
                    'Step count': sum(map(len, ros)),
                    }.items()))
