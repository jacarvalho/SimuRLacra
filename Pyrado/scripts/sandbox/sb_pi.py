"""
Script to test the Planar-Insert environment with the task activation action model
"""
import rcsenv
from pyrado.environments.rcspysim.planar_insert import PlanarInsertSim
from pyrado.plotting.rollout_based import plot_adn_data
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


rcsenv.setLogLevel(0)


def policy_fcn(t: float):
    return [0.7, 1, 0, 0.1, 0.5, 0.5]


if __name__ == '__main__':
    # Set up environment
    dt = 1/1000.
    env = PlanarInsertSim(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gPlanarInsert6Link.xml',  # gPlanarInsert6Link.xml or gPlanarInsert5Link.xml
        dt=dt,
        max_steps=int(10/dt),
        max_dist_force=None,
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
    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate and plot potentials
    print(env.obs_space.labels)
    ro = rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=False)
    plot_adn_data(ro)
