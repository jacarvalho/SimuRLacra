"""
Planned: Test Linear Policy with RBF Features for the WAM Ball-in-a-cup task.
Current: Stabilize WAM arm at his initial position
"""
import torch as to
import numpy as np
import matplotlib.pyplot as plt

from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.policies.base import Policy
from pyrado.utils.data_types import RenderMode, EnvSpec
from pyrado.policies.features import FeatureStack, RBFFeat
from pyrado.policies.linear import LinearPolicy
from pyrado.sampling.rollout import rollout


class DummyMovPrimPolicy(Policy):

    def __init__(self, spec: EnvSpec):
        super().__init__(spec)
        self._curr_step = None

        num_basis_functions = 5
        weights = np.random.normal(0, 10*np.pi/180, (num_basis_functions, 3))
        q, qd = compute_trajectory(weights)
        self.traj = np.concatenate([q, qd], axis=1)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def reset(self):
        self._curr_step = 0

    def forward(self, obs: to.Tensor) -> to.Tensor:
        act = to.tensor(self.traj[self._curr_step])
        self._curr_step += 1
        return act


def compute_trajectory(weights):
    weights = np.concatenate([np.zeros((2, 3)), weights, np.zeros((2, 3))])

    length = 3.5  # time in seconds
    t = np.linspace(0, 1, int(length*500)).reshape(-1, 1)  # Observations [normalized timesteps]
    centers = np.linspace(0, 1, weights.shape[0]).reshape(1, -1)  # RBF center locations
    diffs = t - centers

    width = 0.0035  # pyrado's scale = 1/(2*width)

    # features
    w = np.exp(- diffs ** 2/(2*width))
    wd = - (diffs/width)*w

    w_sum = np.sum(w, axis=1, keepdims=True)
    wd_sum = np.sum(wd, axis=1, keepdims=True)

    # normalized features
    pos_features = w/w_sum
    vel_features = (wd*w_sum - w*wd_sum)/w_sum ** 2

    # trajectory (corresponds to linear policy)
    q = pos_features@weights
    qd = vel_features@weights

    """ Check if gradient is correct with finite difference approximation """
    for i in range(q.shape[1]):
        qd_approx = np.gradient(q[:, i], 1/(length*500))
        assert np.allclose(qd_approx, qd[:, i], rtol=1e-3, atol=1e-3)

    """ Possible Plotting
    plt.figure()
    plt.plot(t, pos_features)
    plt.figure()
    plt.plot(t, vel_features)
    plt.show()
    
    plt.figure()
    plt.plot(t, q)
    plt.figure()
    plt.plot(t, qd)
    plt.show()
    """

    return q, qd


def compute_trajectory_pyrado(weights):
    weights = np.concatenate([np.zeros((2, 3)), weights, np.zeros((2, 3))])
    weights = to.from_numpy(weights)

    length = 3.5
    width = 0.0035
    time = np.linspace(0, 1, int(length*500)).reshape(-1, 1)  # the observations are numpy arrays, so we mimic this here
    time = to.tensor(time, requires_grad=True)  # policy and backprop works with tensors

    rbf = RBFFeat(num_feat_per_dim=weights.shape[0],
                  bounds=(np.array([0.]), np.array([1.])),
                  scale=1/(2*width))

    pos_feat = rbf(time)
    q = pos_feat@weights

    """ Explicit """

    vel_feat_E = rbf.derivative(time)
    qd_E = vel_feat_E@weights

    """ Autograd """

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

    """ Check similarity """
    assert to.norm(qd_E - qd) < 1e-6

    return q, qd


def check_feat_equality():
    weights = np.random.normal(0, 10*np.pi/180, (5, 3))
    q1, qd1 = compute_trajectory_pyrado(weights)
    q2, qd2 = compute_trajectory(weights)

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


def main_stabilization(env):
    env.reset()
    act = np.zeros((6,))
    for i in range(env.max_steps):
        env.step(act)
        env.render(mode=RenderMode(video=True))


def main_dummy_mp(env):
    env.reset()

    # Dummy Movement Primitive Policy
    policy = DummyMovPrimPolicy(env.spec)

    res = rollout(env=env, policy=policy, eval=True, render_mode=RenderMode(video=True))
    des_pos_traj = res.env_infos['des_pos']
    pos_traj = res.env_infos['pos']
    des_vel_traj = res.env_infos['des_vel']
    vel_traj = res.env_infos['vel']

    # Plot trajectories of joints 1 and 3 and their corresponding desired trajectories
    for idx in [1, 3]:
        plt.figure()
        plt.plot(des_pos_traj[:, idx], label=f'des_qpos {idx}')
        plt.plot(pos_traj[:, idx], label=f'qpos {idx}')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    np.random.seed(101)

    # Check for function equality
    print(check_feat_equality())

    # Environment
    env = WAMBallInCupSim(max_steps=1750)
    env.reset()
    env.render(mode=RenderMode(video=True))
    env.viewer._paused = True

    # Stabilize around initial position
    main_stabilization(env)

    # Apply a dummy movement primitive policy
    main_dummy_mp(env)
