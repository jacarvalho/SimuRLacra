import numpy as np
import torch as to
from init_args_serializer.serializable import Serializable

from pyrado.environments.pysim.base import SimPyEnv
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn


class QQubeSim(SimPyEnv, Serializable):
    """
    Environment which models Quanser's Furuta pendulum called Quanser Qube.
    The goal is to swing the pendulum from a hanging down (alpha = 0) into a upright position (alpha = +-pi).
    """

    name: str = 'qq'

    def _create_spaces(self):
        max_volt = self.domain_param['max_volt']

        # Define the spaces
        max_state = np.array([115./180*np.pi, 4*np.pi, 20*np.pi, 20*np.pi])  # [rad, rad, rad/s, rad/s]
        max_obs = np.array([1., 1., 1., 1., np.inf, np.inf])  # [-, -, -, -, rad/s, rad/s]
        max_init_state = np.array([2./180*np.pi, 1./180*np.pi,  # [rad, rad, ...
                                   0.5/180*np.pi, 0.5/180*np.pi])  # ... rad/s, rad/s]

        self._state_space = BoxSpace(-max_state, max_state,
                                     labels=[r'$\theta$', r'$\alpha$', r'$\dot{\theta}$', r'$\dot{\alpha}$'])
        self._obs_space = BoxSpace(-max_obs, max_obs,
                                   labels=[r'$\sin\theta$', r'$\cos\theta$', r'$\sin\alpha$', r'$\cos\alpha$',
                                           r'$\dot{\theta}$', r'$\dot{\alpha}$'])
        self._init_space = BoxSpace(-max_init_state, max_init_state,
                                    labels=[r'$\theta$', r'$\alpha$', r'$\dot{\theta}$', r'$\dot{\alpha}$'])
        self._act_space = BoxSpace(-max_volt, max_volt, labels=['$V$'])

    def _create_task(self, state_des: [np.ndarray, None]) -> Task:
        # Define the task including the reward function
        if state_des is None:
            state_des = np.array([0., np.pi, 0., 0.])
        Q = np.diag([3e-1, 1., 2e-2, 5e-3])
        R = np.diag([4e-3])
        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    @property
    def obs_space(self):
        return self._obs_space

    def observe(self, state, dtype=np.ndarray):
        if dtype is np.ndarray:
            return np.array([np.sin(state[0]), np.cos(state[0]),
                             np.sin(state[1]), np.cos(state[1]),
                             state[2], state[3]])
        elif dtype is to.Tensor:
            return to.cat((state[0].sin().view(1), state[0].cos().view(1),
                           state[1].sin().view(1), state[1].cos().view(1),
                           state[2].view(1), state[3].view(1)))

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(g=9.81,  # gravity [m/s**2]
                    Rm=8.4,  # motor resistance [Ohm]
                    km=0.042,  # motor back-emf constant [V*s/rad]
                    Mr=0.095,  # rotary arm mass [kg]
                    Lr=0.085,  # rotary arm length [m]
                    Dr=5e-6,  # rotary arm viscous damping [N*m*s/rad], original: 0.0015, identified: 5e-6
                    Mp=0.024,  # pendulum link mass [kg]
                    Lp=0.129,  # pendulum link length [m]
                    Dp=1e-6,  # pendulum link viscous damping [N*m*s/rad], original: 0.0005, identified: 1e-6
                    max_volt=4.)  # max action [V]

    def _calc_constants(self):
        Mr = self.domain_param['Mr']
        Mp = self.domain_param['Mp']
        Lr = self.domain_param['Lr']
        Lp = self.domain_param['Lp']
        g = self.domain_param['g']

        # Moments of inertia
        Jr = Mr*Lr ** 2/12  # inertia about COM of the rotary pole [kg*m^2]
        Jp = Mp*Lp ** 2/12  # inertia about COM of the pendulum pole [kg*m^2]

        # Constants for equations of motion
        self._c = np.zeros(5)
        self._c[0] = Jr + Mp*Lr ** 2
        self._c[1] = 0.25*Mp*Lp ** 2
        self._c[2] = 0.5*Mp*Lp*Lr
        self._c[3] = Jp + self._c[1]
        self._c[4] = 0.5*Mp*Lp*g

    def _dyn(self, t, x, u):
        r"""
        Compute $\dot{x} = f(x, u, t)$.

        :param t: time (if the dynamics explicitly depend on the time)
        :param x: state
        :param u: control command
        :return: time derivative of the state
        """
        km = self.domain_param['km']
        Rm = self.domain_param['Rm']
        Dr = self.domain_param['Dr']
        Dp = self.domain_param['Dp']

        # Decompose state
        th, al, thd, ald = x

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1]*np.sin(al) ** 2
        b = self._c[2]*np.cos(al)
        c = self._c[3]
        det = a*c - b*b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = km*(u - km*thd)/Rm
        c0 = self._c[1]*np.sin(2*al)*thd*ald - self._c[2]*np.sin(al)*ald*ald
        c1 = -0.5*self._c[1]*np.sin(2*al)*thd*thd + self._c[4]*np.sin(al)
        x = trq - Dr*thd - c0
        y = -Dp*ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c*x - b*y)/det
        aldd = (a*y - b*x)/det

        return np.array([thd, ald, thdd, aldd])

    def _step_dynamics(self, act: np.ndarray):
        # Compute the derivative
        thd, ald, thdd, aldd = self._dyn(None, self.state, act)

        # Integration step (Runge-Kutta 4)
        k = np.zeros(shape=(4, 4))  # derivatives
        k[0, :] = np.array([thd, ald, thdd, aldd])
        for j in range(1, 4):
            if j <= 2:
                s = self.state + self._dt/2.*k[j - 1, :]
            else:
                s = self.state + self._dt*k[j - 1, :]
            thd, ald, thdd, aldd = self._dyn(None, self.state, act)
            k[j, :] = np.array([s[2], s[3], thdd, aldd])
        self.state += self._dt/6*(k[0] + 2*k[1] + 2*k[2] + k[3])

    def _init_anim(self):
        import vpython as vp

        # Convert to float for VPython
        Lr = float(self.domain_param['Lr'])
        Lp = float(self.domain_param['Lp'])

        # Init render objects on first call
        self._anim['canvas'] = vp.canvas(width=800, height=600, title="Quanser Qube")
        scene_range = 0.2
        arm_radius = 0.003
        pole_radius = 0.0045
        self._anim['canvas'].background = vp.color.white
        self._anim['canvas'].lights = []
        vp.distant_light(direction=vp.vec(0.2, 0.2, 0.5),
                         color=vp.color.white)
        self._anim['canvas'].up = vp.vec(0, 0, 1)
        self._anim['canvas'].range = scene_range
        self._anim['canvas'].center = vp.vec(0.04, 0, 0)
        self._anim['canvas'].forward = vp.vec(-2, 1.2, -1)
        vp.box(pos=vp.vec(0, 0, -0.07), length=0.09, width=0.1, height=0.09, color=vp.color.gray(0.5))
        vp.cylinder(axis=vp.vec(0, 0, -1), radius=0.005, length=0.03, color=vp.color.gray(0.5))
        # Joints
        self._anim['joint1'] = vp.sphere(radius=0.005, color=vp.color.white)
        self._anim['joint2'] = vp.sphere(radius=pole_radius, color=vp.color.white)
        # Arm
        self._anim['arm'] = vp.cylinder(radius=arm_radius, length=Lr, color=vp.color.blue)
        # Pole
        self._anim['pole'] = vp.cylinder(radius=pole_radius, length=Lp, color=vp.color.red)
        # Curve
        self._anim['curve'] = vp.curve(color=vp.color.white, radius=0.0005, retain=2000)

    def _update_anim(self):
        import vpython as vp

        # Convert to float for VPython
        g = self.domain_param['g']
        Mr = self.domain_param['Mr']
        Mp = self.domain_param['Mp']
        Lr = float(self.domain_param['Lr'])
        Lp = float(self.domain_param['Lp'])
        km = self.domain_param['km']
        Rm = self.domain_param['Rm']
        Dr = self.domain_param['Dr']
        Dp = self.domain_param['Dp']

        th, al, _, _ = self.state
        arm_pos = (Lr*np.cos(th), Lr*np.sin(th), 0.)
        pole_ax = (-Lp*np.sin(al)*np.sin(th), +Lp*np.sin(al)*np.cos(th), -Lp*np.cos(al))
        self._anim['arm'].axis = vp.vec(*arm_pos)
        self._anim['pole'].pos = vp.vec(*arm_pos)
        self._anim['pole'].axis = vp.vec(*pole_ax)
        self._anim['joint1'].pos = self._anim['arm'].pos
        self._anim['joint2'].pos = self._anim['pole'].pos
        self._anim['curve'].append(self._anim['pole'].pos + self._anim['pole'].axis)

        # Set caption text
        self._anim['canvas'].caption = f"""
            theta: {self.state[0]*180/np.pi : 3.1f}
            alpha: {self.state[1]*180/np.pi : 3.1f}
            dt: {self._dt :1.4f}
            g: {g : 1.3f}
            Mr: {Mr : 1.4f}
            Mp: {Mp : 1.4f}
            Lr: {Lr : 1.4f}
            Lp: {Lp : 1.4f}
            Dr: {Dr : 1.7f}
            Dp: {Dp : 1.7f}
            Rm: {Rm : 1.3f}
            km: {km : 1.4f}
            """

    def _reset_anim(self):
        # Reset VPython animation
        if self._anim['curve'] is not None:
            self._anim['curve'].clear()


class QQubeStabSim(QQubeSim):
    """
    Environment which models Quanser's Furuta pendulum called Quanser Qube.
    The goal is to stabilize the pendulum at  a upright position (alpha = +-pi).

    .. note::
        This environment is only for testing purposes, or to find the PD gains for stabilizing the pendulum at the top.
    """

    def _create_spaces(self):
        max_volt = self.domain_param['max_volt']

        # Define the spaces
        max_state = np.array([120./180*np.pi, 4*np.pi, 20*np.pi, 20*np.pi])  # [rad, rad, rad/s, rad/s]
        min_init_state = np.array([-5./180*np.pi, 175./180*np.pi, 0, 0])  # [rad, rad, rad/s, rad/s]
        max_init_state = np.array([5./180*np.pi, 185./180*np.pi, 0, 0])  # [rad, rad, rad/s, rad/s]

        self._state_space = BoxSpace(-max_state, max_state,
                                     labels=[r'$\theta$', r'$\alpha$', r'$\dot{\theta}$', r'$\dot{\alpha}$'])
        self._obs_space = self._state_space
        self._init_space = BoxSpace(min_init_state, max_init_state,
                                    labels=[r'$\theta$', r'$\alpha$', r'$\dot{\theta}$', r'$\dot{\alpha}$'])
        self._act_space = BoxSpace(-max_volt, max_volt, labels=['$V$'])

    def observe(self, state, dtype=np.ndarray):
        # Directly observe the noise-free state
        return state.copy()

    def _create_task(self, state_des: [np.ndarray, None]) -> Task:
        # Define the task including the reward function
        if state_des is None:
            state_des = np.array([0., np.pi, 0., 0.])
        Q = np.diag([3., 4., 2., 2.])
        R = np.diag([5e-2])
        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])
