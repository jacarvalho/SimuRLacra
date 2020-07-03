"""
Script to test the Planar-Insert environment with the task activation action model
"""
import math

import rcsenv
from pyrado.environments.rcspysim.planar_insert import PlanarInsertIKSim, PlanarInsertTASim
from pyrado.plotting.rollout_based import plot_potentials
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


rcsenv.setLogLevel(4)


def ik_control_variant(dt, max_steps, max_dist_force, physics_engine, graph_file_name):
    # Set up environment
    env = PlanarInsertIKSim(
        physicsEngine=physics_engine,
        graphFileName=graph_file_name,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeForceTorque=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeGoalDistance=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
    )
    env.reset(domain_param=dict(effector_friction=1.))

    # Set up policy
    def policy_fcn(t: float):
        return [0.1*dt, -0.01*dt, 3/180.*math.pi*math.sin(2.*math.pi*2.*t)]  # [m/s, m/s, rad/s]
    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate and plot potentials
    print(env.obs_space.labels)
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=False)


def task_activation_variant(dt, max_steps, max_dist_force, physics_engine, graph_file_name):
    # Set up environment
    env = PlanarInsertTASim(
        physicsEngine=physics_engine,
        graphFileName=graph_file_name,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        taskCombinationMethod='sum',  # 'sum', 'mean',  'product', or 'softmax'
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeForceTorque=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeGoalDistance=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
    )
    env.reset(domain_param=dict(effector_friction=1.))

    # Set up policy
    def policy_fcn(t: float):
        return [0.7, 1, 0, 0.1, 0.5, 0.5]
    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate and plot potentials
    print(env.obs_space.labels)
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=False)


if __name__ == '__main__':
    # Choose setup
    setup_type = 'ik'  # ik, or activation
    common_hparam = dict(
        dt=0.01,
        max_steps=1200,
        max_dist_force=None,
        physics_engine='Bullet',  # Bullet or Vortex
        graph_file_name='gPlanarInsert6Link.xml',  # gPlanarInsert6Link.xml or gPlanarInsert5Link.xml
    )

    if setup_type == 'ik':
        ro = ik_control_variant(**common_hparam)
    elif setup_type == 'activation':
        ro = task_activation_variant(**common_hparam)
        plot_potentials(ro)
