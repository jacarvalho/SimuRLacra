"""
Test the functionality of Pyrado using the Quanser Ball balancer setup.
"""
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


def policy_fcn(t: float):
    return [
            0.0,  # V_x
            0.0,  # V_y
    ]


# Set up environment
dt = 1 / 500.
env = QBallBalancerSim(dt=dt, max_steps=10000)
env.reset(domain_param=dict(offset_th_x=50. / 180 * 3.141592))
print_domain_params(env.domain_param)

# Set up policy
policy = TimePolicy(env.spec, policy_fcn, dt)

# Simulate
ro = rollout(env, policy, render_mode=RenderMode(text=True, video=True), stop_on_done=True)
