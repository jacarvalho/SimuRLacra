import numpy as np
from init_args_serializer.serializable import Serializable

from pyrado.environments.pysim.base import SimPyEnv
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.reward_functions import QuadrErrRewFcn

# For OneMassOscillatorDyn
import sys
import torch as to
from typing import Sequence
from tqdm import tqdm
from tabulate import tabulate

from pyrado.sampling.rollout import StepSequence
from pyrado.sampling.utils import gen_batches


class OneMassOscillatorSim(SimPyEnv, Serializable):
    """  Model of a linear one-mass-oscillator (spring-mass-damper system) without gravity influence """

    name: str = 'omo'

    def _create_spaces(self):
        k = self.domain_param['k']

        # Define the spaces
        max_state = np.array([1., 10.])  # pos [m], vel [m/s]
        min_init_state = np.array([-0.8*max_state[0], -0.01*max_state[1]])
        max_init_state = np.array([-0.6*max_state[0], +0.01*max_state[1]])
        max_act = np.array([max_state[0]*k])  # max force [N]; should be big enough to reach every steady state
        self._curr_act = np.zeros_like(max_act)  # just for usage in render function

        self._state_space = BoxSpace(-max_state, max_state, labels=['$x$', r'$\dot{x}$'])
        self._obs_space = self._state_space
        self._init_space = BoxSpace(min_init_state, max_init_state, labels=['$x$', r'$\dot{x}$'])
        self._act_space = BoxSpace(-max_act, max_act, labels=['$F$'])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', None)
        if state_des is None:
            state_des = np.zeros(2)
        Q = np.diag([1e1, 1e-3])
        R = np.diag([1e-6])
        return FinalRewTask(
            DesStateTask(self.spec, state_des, QuadrErrRewFcn(Q, R)), factor=1e3, mode=FinalRewMode(always_negative=True)
        )

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(m=1.,  # object's mass [kg]
                    k=30.,  # spring stiffness constant [N/m]
                    d=0.5)  # damping constant [Ns/m]

    def _calc_constants(self):
        m = self.domain_param['m']
        k = self.domain_param['k']
        d = self.domain_param['d']

        self.omega = np.sqrt(k/m)  # eigen frequency [Hz]
        self.zeta = d/(2.*np.sqrt(m*k))  # damping ratio [-]
        if self.zeta < 1.:
            self._omega_d = np.sqrt(1 - self.zeta**2)*self.omega  # damped eigen frequency [Hz]
        else:
            self._omega_d = None  # overdamped case, no oscillation
        if self.zeta < np.sqrt(1/2):
            self._omega_res = np.sqrt(1 - 2.*self.zeta**2)*self.omega  # resonance frequency [Hz]
        else:
            self._omega_res = None  # damping too high, no resonance

    def _step_dynamics(self, act: np.ndarray):
        m = self.domain_param['m']

        # Linear Dynamics
        A = np.array([[0, 1], [-self.omega**2, -2.*self.zeta*self.omega]])
        B = np.array([[0], [1./m]])
        state_dot = A.dot(self.state) + B.dot(act).reshape(2, )

        # Integration Step (forward Euler)
        self.state = self.state + state_dot*self._dt  # next state

    def _init_anim(self):
        import vpython as vp
        c = 0.1*self.obs_space.bound_up[0]

        self._anim['canvas'] = vp.canvas(width=1000, height=400, title="One Mass Oscillator")
        self._anim['ground'] = vp.box(
            pos=vp.vec(0, -0.02, 0),
            length=2.*self.obs_space.bound_up[0],
            height=0.02,
            width=3*c,
            color=vp.color.green,
            canvas=self._anim['canvas']
        )
        self._anim['mass'] = vp.box(
            pos=vp.vec(self.state[0], c/2., 0),
            length=c,
            height=c,
            width=c,
            color=vp.color.blue,
            canvas=self._anim['canvas']
        )
        self._anim['des'] = vp.box(
            pos=vp.vec(self._task.state_des[0], 0.8*c/2., 0),
            length=0.8*c,
            height=0.8*c,
            width=0.8*c,
            color=vp.color.cyan,
            opacity=0.5,  # 0 is fully transparent
            canvas=self._anim['canvas']
        )
        self._anim['force'] = vp.arrow(
            pos=vp.vec(self.state[0], c/2., 0),
            axis=vp.vec(0.1*self._curr_act, 0, 0),
            color=vp.color.red,
            shaftwidth=0.2*c,
            canvas=self._anim['canvas']
        )
        self._anim['spring'] = vp.helix(
            pos=vp.vec(0, c/2., 0),
            axis=vp.vec(self.state[0] - c/2., 0, 0),
            color=vp.color.blue,
            radius=c/3.,
            canvas=self._anim['canvas']
        )

    def _update_anim(self):
        import vpython as vp
        m = self.domain_param['m']
        k = self.domain_param['k']
        d = self.domain_param['d']
        c = 0.1*self.obs_space.bound_up[0]

        self._anim['mass'].pos = vp.vec(self.state[0], c/2., 0)
        self._anim['force'].pos = vp.vec(self.state[0], c/2., 0)
        capped_act = np.sign(self._curr_act)*np.max((0.1*np.abs(self._curr_act), 0.3))
        self._anim['force'].axis = vp.vec(capped_act, 0, 0)
        self._anim['spring'].axis = vp.vec(self.state[0] - c/2., 0., 0)

        # Set caption text
        self._anim['canvas'].caption = f"""
            dt: {self.dt :1.4f}
            m: {m : 1.3f}
            k: {k : 2.2f}
            d: {d : 1.3f}
            """

    def _reset_anim(self):
        import vpython as vp
        c = 0.1*self.obs_space.bound_up[0]
        self._anim['mass'].pos = vp.vec(self.state[0], c/2., 0)
        self._anim['des'].pos = vp.vec(self._task.state_des[0], 0.8*c/2., 0)
        self._anim['force'].pos = vp.vec(self.state[0], c/2., 0)
        self._anim['force'].axis = vp.vec(0.1*self._curr_act, 0, 0)
        self._anim['spring'].axis = vp.vec(self.state[0] - c/2., 0., 0)


class OneMassOscillatorDyn(Serializable):

    def __init__(self, dt: float):
        """
        Constructor

        :param dt: simulation step size [s]
        """
        Serializable._init(self, locals())

        self._dt = dt
        self.omega = None
        self.zeta = None
        self.A = None
        self.B = None

    def _calc_constants(self, dp: dict):
        """
        Calculate the physics constants that depend on the domain parameters.

        :param dp: current domain parameter estimate
        """
        self.omega = to.sqrt(dp['k']/dp['m'])
        self.zeta = dp['d']/(2.*to.sqrt(dp['m']*dp['k']))

        self.A = to.stack([to.tensor([0., 1.]), to.stack([-self.omega**2, -2.*self.zeta*self.omega])])
        self.B = to.stack([to.tensor(0.), 1./dp['m']]).view(-1, 1)

    def __call__(self, state: to.Tensor, act: to.Tensor, domain_param: dict) -> to.Tensor:
        """
        One step of the forward dynamics

        :param state: current state
        :param act: current action
        :param domain_param: current domain parameters
        :return: next state
        """
        self._calc_constants(domain_param)
        # act = self._limit_act(act)

        # state_dot = self.A @ state + self.B @ act
        state_dot = state@self.A.t() + act@self.B.t()  # Pyro batching

        # Return the state delta (1st order approximation)
        return state_dot*self._dt


class OneMassOscillatorDomainParamEstimator(to.nn.Module):
    """ Class to estimate the domain parameters of the OneMassOscillator environment """

    def __init__(self, dt: float, dp_init: dict, num_epoch: int, batch_size: int):
        super().__init__()

        self.dp_est = to.nn.Parameter(to.tensor([dp_init['m'], dp_init['k'], dp_init['d']]), requires_grad=True)
        self.dp_fixed = dict(dt=dt)

        self.optim = to.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        self.loss_fcn = to.nn.MSELoss()
        self.num_epoch = num_epoch
        self.batch_size = batch_size

        # Define the dynamics
        self.dyn = OneMassOscillatorDyn(dt)

    def forward(self, state: to.Tensor, act: to.Tensor) -> to.Tensor:
        return self.dyn(state, act, dict(m=self.dp_est[0], k=self.dp_est[1], d=self.dp_est[2]))

    def update(self, rollouts: Sequence[StepSequence]):
        # Pre-process rollout data
        [ro.torch(data_type=to.get_default_dtype()) for ro in rollouts]
        states_cat = to.cat([ro.observations[:-1] for ro in rollouts])
        actions_cat = to.cat([ro.actions for ro in rollouts])
        targets_cat = to.cat([(ro.observations[1:] - ro.observations[:-1]) for ro in rollouts])  # state deltas

        # Iteration over the full data set
        for e in range(self.num_epoch):
            loss_list = []

            # Mini-batch optimization
            for idcs in tqdm(gen_batches(self.batch_size, len(targets_cat)),
                             total=(len(targets_cat) + self.batch_size - 1)//self.batch_size,
                             desc=f'Epoch {e}', unit='batches', file=sys.stdout, leave=False):
                # Make predictions
                preds = to.stack([self.forward(s, a) for s, a in zip(states_cat[idcs], actions_cat[idcs])])

                # Reset the gradients and call the optimizer
                self.optim.zero_grad()
                loss = self.loss_fcn(preds, targets_cat[idcs])
                loss.backward()
                loss_list.append(loss.detach().numpy())
                self.optim.step()

            print(tabulate([['avg loss', np.mean(loss_list)],
                            ['param estimate', self.dp_est.cpu().detach().numpy()]]))
