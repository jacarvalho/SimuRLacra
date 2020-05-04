"""
Test the wrapping of an Isaac Gym environment.
"""
from pyrado.environments.isaac_gym.base import IsaacSimEnv
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Set up environment
    env = IsaacSimEnv(dt=1/50., task_args=None, max_steps=100)

    # Set up policy
    policy = IdlePolicy(env.spec)
    # Simulate
    done, param, state = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=False, video=True), eval=True,
                     reset_kwargs=dict(domain_param=param, init_state=state))
        print_domain_params(env.domain_param)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, ro)
