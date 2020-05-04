"""
Basic script to test the bi-manual activation based task using a hard-coded time-based policy
"""
import numpy as np
import os.path as osp
from pyrado.environments.rcspysim.target_tracking import TargetTrackingSim

import rcsenv
from pyrado.plotting.rollout_based import plot_rewards
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout


# rcsenv.setLogLevel(7)
from pyrado.utils.data_types import RenderMode


def policy_fcn(t: float):
    if t > 2:
        return [0., 1., 0.5]
    else:
        return [1., 0., 1.]


# Set up environment
dt = 0.01
env = TargetTrackingSim(
    physicsEngine='Bullet',  # Bullet or Vortex
    graphFileName='TargetTracking.xml',
    dt=dt,
    max_steps=1000,
    mps_left=[
        {
            'function': 'lin',
            'errorDynamics': 1. * np.eye(3),
            'goal': np.array([0.5, 0.25, 1]),
        },
        {
            'function': 'lin',
            'errorDynamics': 1. * np.eye(3),
            'goal': np.array([0.5, -0.5, 1]),
        },
    ],
    mps_right=[
        {
            'function': 'lin',
            'errorDynamics': 1. * np.eye(3),
            'goal': np.array([0.25, 0.25, 1]),
        },
    ],
    collisionConfig={
        'file': "collisionModel.xml",
        # 'pairs': [
        #     {
        #         'body1': 'PowerGrasp_L',
        #         'body2': 'PowerGrasp_R',
        #     },
        # ],
        # 'threshold': 0.1,
    },
    checkJointLimits=True,
)
print(env.obs_space.labels)

# Set up policy
policy = TimePolicy(spec=env.spec, fcn_of_time=policy_fcn, dt=dt)

# Create scripted version of the policy
tm = policy.trace()
print(tm.graph)
print(tm.code)

# Export config and policy for C++
env.save_config_xml(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'TargetTracking', "exTT_export.xml"))
tm.save(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'TargetTracking', 'pTT_simpletime.pth'))

# Simulate and plot
ro = rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)
plot_rewards(ro)
