"""
Test predefined energy-based controller to make the Quanser Qube swing up.
"""
import torch as to

from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Set up environment
    env = QQubeSim(dt=1/500., max_steps=4000)

    # Set up policy
    policy = QQubeSwingUpAndBalanceCtrl(env.spec)

    # Simulate
    done, param, state = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=False, video=True), eval=True,
                     reset_kwargs=dict(domain_param=param, init_state=state))
        print_domain_params(env.domain_param)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, ro)
