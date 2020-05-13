"""
Test MuJoCo-based Hopper environment with a random policy.
"""
from pyrado.environments.mujoco.openai_hopper import HopperSim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.dummy import DummyPolicy, IdlePolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Set up environment
    env = HopperSim()

    # Set up policy
    # policy = DummyPolicy(env.spec)
    policy = IdlePolicy(env.spec)

    # Simulate
    done, param, state = False, None, None
    while not done:
        env.reset()
        print_cbt(f'init obs (before): {env.observe(env.state)}', 'c')
        ro = rollout(env, policy, render_mode=RenderMode(text=False, video=True), eval=True,
                     reset_kwargs=dict(domain_param=param, init_state=env.state.copy()))
        print_domain_params(env.domain_param)
        print_cbt(f'init obs (after): {ro.observations[0]}', 'c')
        done, state, param = after_rollout_query(env, policy, ro)
