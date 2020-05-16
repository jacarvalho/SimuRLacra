"""
Test Linear Policy with RBF Features for the WAM Ball-in-a-cup task.
"""
import torch as to
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

import pyrado
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.policies.environment_specific import DualRBFLinearPolicy
from pyrado.utils.data_types import RenderMode
from pyrado.policies.features import RBFFeat
from pyrado.sampling.rollout import rollout, after_rollout_query


def compute_trajectory(weights, time, width):
    centers = np.linspace(0, 1, weights.shape[0]).reshape(1, -1)  # RBF center locations
    diffs = time - centers

    # Features
    w = np.exp(- diffs ** 2/(2*width))
    wd = - (diffs/width)*w

    w_sum = np.sum(w, axis=1, keepdims=True)
    wd_sum = np.sum(wd, axis=1, keepdims=True)

    # Normalized features
    pos_features = w/w_sum
    vel_features = (wd*w_sum - w*wd_sum)/w_sum ** 2

    # Trajectory
    q = pos_features@weights
    qd = vel_features@weights

    # Check gradient computation with finite difference approximation
    for i in range(q.shape[1]):
        qd_approx = np.gradient(q[:, i], 1/len(time))
        assert np.allclose(qd_approx, qd[:, i], rtol=1e-3, atol=1e-3)

    return q, qd


def compute_trajectory_pyrado(weights, time, width):
    weights = to.from_numpy(weights)
    time = to.tensor(time, requires_grad=True)
    rbf = RBFFeat(num_feat_per_dim=weights.shape[0],
                  bounds=(np.array([0.]), np.array([1.])),
                  scale=1/(2*width))
    pos_feat = rbf(time)
    q = pos_feat@weights

    # Explicit
    vel_feat_E = rbf.derivative(time)
    qd_E = vel_feat_E@weights

    # Autograd
    q1, q2, q3 = q.t()
    q1.backward(to.ones((1750,)), retain_graph=True)
    q1d = time.grad.clone()
    time.grad.fill_(0.)
    q2.backward(to.ones((1750,)), retain_graph=True)
    q2d = time.grad.clone()
    time.grad.fill_(0.)
    q3.backward(to.ones((1750,)))
    q3d = time.grad.clone()
    qd = to.cat([q1d, q2d, q3d], dim=1)

    # Check similarity
    assert to.norm(qd_E - qd) < 1e-6

    return q, qd


def check_feat_equality():
    weights = np.random.normal(0, 1, (5, 3))
    time = np.linspace(0, 1, 1750).reshape(-1, 1)
    width = 0.0035
    q1, qd1 = compute_trajectory_pyrado(weights, time, width)
    q2, qd2 = compute_trajectory(weights, time, width)

    assert q1.size() == q2.shape
    assert qd1.size() == qd2.shape

    is_q_equal = np.allclose(q1.detach().numpy(), q2)
    is_qd_equal = np.allclose(qd1.detach().numpy(), qd2)

    correct = is_q_equal and is_qd_equal

    if not correct:
        _, axs = plt.subplots(2)
        axs[0].set_title('positions - solid: pyrado, dashed: reference')
        axs[0].plot(q1.detach().numpy())
        axs[0].set_prop_cycle(None)
        axs[0].plot(q2, ls='--')
        axs[1].set_title('velocities - solid: pyrado, dashed: reference, dotted: finite difference')
        axs[1].plot(qd1.detach().numpy())
        axs[1].set_prop_cycle(None)
        axs[1].plot(qd2, ls='--')
        if is_q_equal:  # q1 and a2 are the same
            finite_diff = np.diff(np.concatenate([np.zeros((1, 3)), q2], axis=0)*500., axis=0)  # init with 0, 500Hz
            axs[1].plot(finite_diff, c='k', ls=':')
        plt.show()

    return correct


if __name__ == '__main__':
    # Fix seed for reproducibility
    pyrado.set_seed(101)

    # Check for function equality
    print(check_feat_equality())

    # Environment
    env = WAMBallInCupSim(max_steps=1750)

    # Stabilize around initial position
    env.reset()
    act = np.zeros((6,))  # desired deltas from the initial pose
    for i in range(env.max_steps):
        env.step(act)
        env.render(mode=RenderMode(video=True))

    # Apply DualRBFLinearPolicy
    rbf_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.]), np.array([1.])), scale=None)
    policy = DualRBFLinearPolicy(env.spec, rbf_hparam)
    ro = rollout(env, policy, render_mode=RenderMode(video=True), eval=True)
    after_rollout_query(env, policy, ro)

    # Retrieve infos from rollout
    t = ro.env_infos['t']
    des_pos_traj = ro.env_infos['des_qpos']
    pos_traj = ro.env_infos['qpos']
    des_vel_traj = ro.env_infos['des_qvel']
    vel_traj = ro.env_infos['qvel']
    ball_pos = ro.env_infos['ball_pos']
    state_des = ro.env_infos['state_des']

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(3, sharex='all')
    for i, idx in enumerate([1, 3, 5]):
        ax[i].plot(t, des_pos_traj[:, idx], label=f'des_qpos {idx}')
        ax[i].plot(t, pos_traj[:, idx], label=f'qpos {idx}')
        ax[i].legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs=ball_pos[:, 0], ys=ball_pos[:, 1], zs=ball_pos[:, 2], color='blue', label='Ball')
    ax.scatter(xs=ball_pos[-1, 0], ys=ball_pos[-1, 1], zs=ball_pos[-1, 2], color='blue', label='Ball Final')
    ax.plot(xs=state_des[:, 0], ys=state_des[:, 1], zs=state_des[:, 2], color='red', label='Cup')
    ax.scatter(xs=state_des[-1, 0], ys=state_des[-1, 1], zs=state_des[-1, 2], color='red', label='Cup Final')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=16., azim=-7.)
    plt.show()
