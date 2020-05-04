"""
Script to test the BoxFiddle environment with different action models
"""
import math
import numpy as np

import rcsenv
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.planar_box_fiddle import PlanarBoxFiddleIKSim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.plotting.rollout_based import plot_adn_data, plot_observations, plot_rewards, plot_actions
from pyrado.policies.adn import ADNPolicy
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode

rcsenv.setLogLevel(0)


def task_activation_variant(dt, max_steps, max_dist_force, physics_engine, graph_file_name):
    # Define a policy
    def policy_fcn(t: float):
        return [0, 0, 0] if t < 6.0 else [0, 0, 0]

    # Set up environment
    env = PlanarBoxFiddleIKSim(
        physicsEngine=physics_engine,  # Bullet or Vortex
        graphFileName=graph_file_name,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
    )
    print_domain_params(env.domain_param)

    # Set up policy
    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)


if __name__ == '__main__':
    # Test the env
    graph_file_name = 'gBoxFiddle.xml'  
    ro = task_activation_variant(dt=0.01, max_steps=1000, max_dist_force=None, physics_engine='Bullet',
                                 graph_file_name=graph_file_name)
    # Plot
    if graph_file_name == 'gBoxFiddle.xml':
        # plot_rewards(ro)
        plot_actions(ro)

