import numpy as np
from init_args_serializer.serializable import Serializable

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.goalless import OptimProxyTask
from pyrado.tasks.reward_functions import StateBasedRewFcn
from pyrado.utils.data_types import RenderMode
from pyrado.utils.functions import rosenbrock


class RosenSim(SimEnv, Serializable):
    """ This environment wraps the Rosenbrock function to use it as a test case for Pyrado algorithms. """

    name: str = 'rosen'

    def __init__(self):
        """ Constructor """
        Serializable._init(self, locals())

        # Initialize basic variables
        super().__init__(dt=None, max_steps=1)

        # Set the bounds for the system's states adn actions
        max_state = np.array([100., 100.])
        max_act = max_state
        self._curr_act = np.zeros_like(max_act)  # just for usage in render function

        self._state_space = BoxSpace(-max_state, max_state, labels=['$x_1$', '$x_2$'])
        self._init_space = SingularStateSpace(np.zeros(self._state_space.shape),
                                              labels=['$x_1_{init}$', '$x_2_{init}$'])
        self._act_space = BoxSpace(-max_act, max_act, labels=['$x_1_{next}$', '$x_2_{next}$'])

        # Define the task including the reward function
        self._task = self._create_task()

        # Animation with pyplot
        self._anim = dict(fig=None, trace_x=[], trace_y=[], trace_z=[])

    @property
    def state_space(self):
        return self._state_space

    @property
    def obs_space(self):
        return self._state_space

    @property
    def init_space(self):
        return self._init_space

    @property
    def act_space(self):
        return self._act_space

    def _create_task(self, task_args: dict = None) -> OptimProxyTask:
        return OptimProxyTask(self.spec, StateBasedRewFcn(rosenbrock, flip_sign=True))

    @property
    def task(self) -> OptimProxyTask:
        return self._task

    @property
    def domain_param(self):
        return {}

    @domain_param.setter
    def domain_param(self, param):
        pass

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return {}

    def reset(self, init_state=None, domain_param=None):
        # Reset time
        self._curr_step = 0

        # Reset the domain parameters
        if domain_param is not None:
            self.domain_param = domain_param

        # Reset the state
        if init_state is None:
            self.state = self._init_space.sample_uniform()  # zero
        else:
            if not init_state.shape == self.obs_space.shape:
                raise pyrado.ShapeErr(given=init_state, expected_match=self.obs_space)
            if isinstance(init_state, np.ndarray):
                self.state = init_state.copy()
            else:
                try:
                    self.state = np.array(init_state)
                except Exception:
                    raise pyrado.TypeErr(given=init_state, expected_type=[np.ndarray, list])

        # No need to reset the task

        # Return perfect observation
        return self.observe(self.state)

    def step(self, act):
        # Apply actuator limits
        act = self._limit_act(act)

        # Action equal selection a new state a.k.a. solution of the optimization problem
        self.state = act

        # Current reward depending on the state after the step (since there is only one step)
        self._curr_rew = self.task.step_rew(self.state)

        self._curr_step += 1

        # Check if the task or the environment is done
        done = self.task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        return self.observe(self.state), self._curr_rew, done, {}

    def render(self, mode: RenderMode, render_step: int = 1):
        # Call base class
        super().render(mode)

        # Print to console
        if mode.text:
            if self._curr_step%render_step == 0 and self._curr_step > 0:  # skip the render before the first step
                print("step: {:3}  |  r: {:1.3f}  |  a: {}  |  s_t+1: {}".format(
                    self._curr_step,
                    self._curr_rew,
                    self._curr_act,
                    self.state)
                )

        # Render using pyplot
        if mode.video:
            from matplotlib import pyplot as plt
            from pyrado.plotting.surface import render_surface

            plt.ion()

            if self._anim['fig'] is None:
                # Plot Rosenbrock function once if not already plotted
                x = np.linspace(-2, 2, 20, True)
                y = np.linspace(-1, 3, 20, True)
                self._anim['fig'] = render_surface(x, y, rosenbrock, 'x', 'y', 'z')

            self._anim['trace_x'].append(self.state[0])
            self._anim['trace_y'].append(self.state[1])
            self._anim['trace_z'].append(rosenbrock(self.state))

            ax = self._anim['fig'].gca()
            ax.scatter(self._anim['trace_x'], self._anim['trace_y'], self._anim['trace_z'], s=8, c='w')

            plt.draw()
