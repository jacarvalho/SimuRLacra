"""
Script to test the Planar-3-Link environment with different action models
"""
import math
import numpy as np
import torch as to

import rcsenv
import pyrado
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.planar_3_link import Planar3LinkJointCtrlSim, Planar3LinkIKSim, Planar3LinkTASim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.plotting.rollout_based import plot_adn_data, plot_rewards
from pyrado.policies.adn import ADNPolicy, pd_cubic
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt

rcsenv.setLogLevel(0)


def joint_control_variant(dt, max_steps, max_dist_force, physics_engine):
    # Set up environment
    env = Planar3LinkJointCtrlSim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        checkJointLimits=True,
    )
    print_domain_params(env.domain_param)

    # Set up policy
    def policy_fcn(t: float):
        return [0.1, 0.1,  # same as init config
                0.1 + 45./180.*math.pi*math.sin(2.*math.pi*0.2*t)]  # oscillation in last link

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)


def ik_control_variant(dt, max_steps, max_dist_force, physics_engine):
    # Set up environment
    env = Planar3LinkIKSim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        checkJointLimits=True,
    )
    print_domain_params(env.domain_param)

    # Set up policy
    def policy_fcn(t: float):
        return [0.3 + 0.2*math.sin(2.*math.pi*0.2*t),
                1.1]

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)


def task_activation_variant(dt, max_steps, max_dist_force, physics_engine):
    # Set up environment
    env = Planar3LinkTASim(
        physicsEngine=physics_engine,
        dt=dt,
        position_mps=True,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeCollisionCost=True,
        observeDynamicalSystemDiscrepancy=False,
    )
    print(env.obs_space.labels)

    # Set up policy
    def policy_fcn(t: float):
        if t < 3:
            return [0, 1, 0]
        elif t < 7:
            return [1, 0, 0]
        elif t < 10:
            return [.5, 0.5, 0]
        else:
            return [0, 0, 1]

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=False)


def task_activation_manual(dt, max_steps, max_dist_force, physics_engine):
    # Set up environment
    env = Planar3LinkTASim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force
    )
    print_domain_params(env.domain_param)

    # Set up policy
    def policy_fcn(t: float):
        pot = np.fromstring(input("Enter potentials for next step: "), dtype=np.double, count=3, sep=' ')
        return 1/(1 + np.exp(-pot))

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)


def adn_variant(dt, max_steps, max_dist_force, physics_engine, normalize_obs=True, obsnorm_cpp=True):
    pyrado.set_seed(1001)

    # Explicit normalization bounds
    elb = {
        'EffectorLoadCell_Fx': -100.,
        'EffectorLoadCell_Fz': -100.,
        'Effector_Xd': -1,
        'Effector_Zd': -1,
        'GD_DS0d': -1,
        'GD_DS1d': -1,
        'GD_DS2d': -1,
    }
    eub = {
        'GD_DS0': 3.,
        'GD_DS1': 3,
        'GD_DS2': 3,
        'EffectorLoadCell_Fx': 100.,
        'EffectorLoadCell_Fz': 100.,
        'Effector_Xd': .5,
        'Effector_Zd': .5,
        'GD_DS0d': .5,
        'GD_DS1d': .5,
        'GD_DS2d': .5,
        'PredCollCost_h50': 1000.
    }

    extra_kwargs = {}
    if normalize_obs and obsnorm_cpp:
        extra_kwargs['normalizeObservations'] = True
        extra_kwargs['obsNormOverrideLower'] = elb
        extra_kwargs['obsNormOverrideUpper'] = eub

    # Set up environment
    env = Planar3LinkTASim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        collisionAvoidanceIK=True,
        taskCombinationMethod='sum',
        **extra_kwargs
    )

    if normalize_obs and not obsnorm_cpp:
        env = ObsNormWrapper(env, explicit_lb=elb, explicit_ub=eub)

    # Set up random policy
    policy_hparam = dict(
        tau_init=0.2,
        output_nonlin=to.sigmoid,
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, dt=dt, **policy_hparam)
    print_cbt('Running ADNPolicy with random initialization', 'c', bright=True)

    # Simulate and plot potentials
    ro = rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)
    plot_adn_data(ro)

    return ro


if __name__ == '__main__':
    # Choose setup
    setup_type = 'ik'  # joint, ik, activation, manual, adn
    common_hparam = dict(
        dt=0.01,
        max_steps=1200,
        max_dist_force=None,
        physics_engine='Bullet',  # Bullet or Vortex
    )

    if setup_type == 'joint':
        ro = joint_control_variant(**common_hparam)
    elif setup_type == 'ik':
        ro = ik_control_variant(**common_hparam)
    elif setup_type == 'activation':
        ro = task_activation_variant(**common_hparam)
        plot_rewards(ro)
        # plt.show()
    elif setup_type == 'manual':
        ro = task_activation_manual(**common_hparam)
    elif setup_type == 'adn':
        ro = adn_variant(**common_hparam, normalize_obs=True, obsnorm_cpp=False)
    else:
        raise pyrado.ValueErr(given=setup_type, eq_constraint="'joint', 'ik', 'activation', 'manual', 'adn'")
