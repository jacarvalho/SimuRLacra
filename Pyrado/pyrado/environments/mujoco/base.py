import mujoco_py
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode


class MujocoSimEnv(SimEnv, ABC, Serializable):
    """
    Base class for MuJoCo environments.
    Uses Serializable to facilitate proper serialization.

    .. seealso::
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py
    """

    def __init__(self,
                 model_path: str,
                 frame_skip: int = 1,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None):
        """
        Constructor

        :param model_path: model path
        :param frame_skip: number of frame skips
        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction, e.g `dict(fwd_rew_weight=1.)`
        """
        Serializable._init(self, locals())

        # Initialize domain parameters and MuJoCo model
        self.model_path = model_path
        self.frame_skip = frame_skip
        self._domain_param = self.get_nominal_domain_param()
        with open(self.model_path, mode='r') as file_raw:
            # Save raw (with placeholders) XML-file as attribute since we need it for resetting the domain params
            self.xml_model_template = file_raw.read()
        self._create_mujoco_model()

        # Call SimEnv's constructor
        super().__init__(dt=self.model.opt.timestep*self.frame_skip, max_steps=max_steps)

        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()

        # Initialize spaces
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._init_space = None
        self._create_spaces()

        # Create task
        if not (isinstance(task_args, dict) or task_args is None):
            raise pyrado.TypeErr(given=task_args, expected_type=dict)
        self.task_args = dict() if task_args is None else task_args
        self._task = self._create_task(self.task_args)

        # Visualization
        self.camera_config = dict()
        self.viewer = None
        self._curr_act = np.zeros(self.act_space.shape)

    @property
    def state_space(self) -> Space:
        return self._state_space

    @property
    def obs_space(self) -> Space:
        return self._obs_space

    @property
    def init_space(self) -> Space:
        return self._init_space

    @property
    def act_space(self) -> Space:
        return self._act_space

    @abstractmethod
    def _create_spaces(self):
        """
        Create spaces based on the domain parameters.
        Should set the attributes `_state_space`, `_act_space`, `_obs_space`, and `_init_space`.

        .. note::
            This function is called from the constructor and from the domain parameter setter.
        """

    @property
    def task(self) -> Task:
        return self._task

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        # Needs to implemented by subclasses
        raise NotImplementedError

    @property
    def domain_param(self) -> dict:
        return deepcopy(self._domain_param)

    @domain_param.setter
    def domain_param(self, param: dict):
        if not isinstance(param, dict):
            raise pyrado.TypeErr(given=param, expected_type=dict)
        # Update the parameters
        self._domain_param.update(param)

        # Update MuJoCo model
        self._create_mujoco_model()

        if self.viewer is not None:
            # If the viewer already exists and we reset the domain parameters, we must also recreate the viewer since
            # it references to the simulation object which get's reconstructed during _create_mujoco_model()
            import glfw
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

        # Update spaces
        self._create_spaces()

        # Update task
        self._task = self._create_task(self.task_args)

    def _adapt_model_file(self, xml_model: str, domain_param: dict) -> str:
        """
        Changes the model's XML-file given the current domain parameters before constructing the MuJoCo simulation.
        One use case is for example the cup_scale for the `WAMBallInCupSim` where multiple values in the model's
        XML-file are changed based on one domain parameter.

        :param xml_model: parsed model file
        :param domain_param: copy of the environments domain parameters
        :return: adapted model file where the placeholders are filled with numerical values
        """
        # The mesh dir is not resolved when later passed as a string, thus we do it manually
        xml_model = xml_model.replace(f'[ASSETS_DIR]', pyrado.MUJOCO_ASSETS_DIR)

        # Replace all occurrences of the domain parameter placeholder with its value
        for key, value in domain_param.items():
            xml_model = xml_model.replace(f'[{key}]', str(value))

        return xml_model

    @abstractmethod
    def _mujoco_step(self, act: np.ndarray) -> dict:
        """
        Apply the given action to the MuJoCo simulation. This executes one step of the physics simulation.

        :param act: action
        :return: `dict` with optional information from MuJoCo
        """

    def _create_mujoco_model(self):
        """
        Called to update the MuJoCo model by rewriting and reloading the XML file.

        .. note::
            This function is called from the constructor and from the domain parameter setter.
        """
        xml_model = self.xml_model_template  # don't change the template
        xml_model = self._adapt_model_file(xml_model, self.domain_param)

        # Create MuJoCo model from parsed XML file
        self.model = mujoco_py.load_model_from_xml(xml_model)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=self.frame_skip)

    def configure_viewer(self):
        """ Configure the camera when the viewer is initialized. You need to set `self.camera_config` before. """
        for key, value in self.camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset time
        self._curr_step = 0

        # Reset the domain parameters
        if domain_param is not None:
            self.domain_param = domain_param

        if init_state is None:
            # Sample init state from init state space
            init_state = self.init_space.sample_uniform()
        elif not isinstance(init_state, np.ndarray):
            # Make sure init state is a numpy array
            try:
                init_state = np.array(init_state)
            except Exception:
                raise pyrado.TypeErr(given=init_state, expected_type=[np.ndarray, list])

        if not self.init_space.contains(init_state, verbose=True):
            pyrado.ValueErr(msg='The init state must be within init state space!')

        # Update the state attribute
        self.state = init_state.copy()

        # Reset the task which also resets the reward function if necessary
        self._task.reset(env_spec=self.spec, init_state=init_state.copy())

        # Reset MuJoCo simulation model (only reset the joint configuration)
        self.sim.reset()
        old_state = self.sim.get_state()
        nq = self.init_qpos.size
        if not init_state[:nq].shape == old_state.qpos.shape:  # check joint positions dimension
            raise pyrado.ShapeErr(given=init_state[:nq], expected_match=old_state.qpos)
        # Exclude everything that is added to the state (at the end), e.g. the ball position for WAMBallInCupSim
        if not init_state[nq:2*nq].shape == old_state.qvel.shape:  # check joint velocities dimension
            raise pyrado.ShapeErr(given=init_state[nq:2*nq], expected_match=old_state.qvel)
        new_state = mujoco_py.MjSimState(
            # Exclude everything that is added to the state (at the end), e.g. the ball position for WAMBallInCupSim
            old_state.time, init_state[:nq], init_state[nq:2*nq], old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

        # Return an observation
        return self.observe(self.state)

    def step(self, act: np.ndarray) -> tuple:
        # Current reward depending on the state (before step) and the (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self.task.step_rew(self.state, act, remaining_steps)

        # Apply actuator limits
        act = self.limit_act(act)
        self._curr_act = act  # just for the render function

        # Apply the action and simulate the resulting dynamics
        info = self._mujoco_step(act)
        info['t'] = self._curr_step*self._dt
        self._curr_step += 1

        # Check if the environment is done due to a failure within the mujoco simulation (e.g. bad inputs)
        mjsim_done = info.get('failed', False)

        # Check if the task is done
        task_done = self._task.is_done(self.state)

        # Handle done case
        done = mjsim_done or task_done
        if self._curr_step >= self._max_steps:
            done = True

        if done:
            # Add final reward if done
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info

    def render(self, mode: RenderMode = RenderMode(), render_step: int = 1):
        if self._curr_step%render_step == 0:
            # Call base class
            super().render(mode)

            # Print to console
            if mode.text:
                print("step: {:3}  |  r: {:1.3f}  |  a: {}  |  s_t+1: {}".format(
                    self._curr_step,
                    self._curr_rew,
                    self._curr_act,
                    self.state))

            # Forward to MuJoCo viewer
            if mode.video:
                if self.viewer is None:
                    # Create viewer if not existent (see 'human' mode of OpenAI Gym's MujocoEnv)
                    self.viewer = mujoco_py.MjViewer(self.sim)
                    self.configure_viewer()
                self.viewer.render()
