import numpy as np
from abc import abstractmethod
from carbongym import gymapi
from init_args_serializer import Serializable
from typing import Iterable, Tuple

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import MinusOnePerStepRewFcn
from pyrado.utils.data_types import RenderMode
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.input_output import print_cbt


class IsaacMasterSimEnv(SimEnv, Serializable):
    """
    Base class for simulated environments implemented in Nvidia's Isaac Gym

    .. note::
         Isaac Gym uses a y-up coordinate convention. They really have no mercy :)
    """

    def __init__(self,
                 num_envs: int,
                 dt: float,
                 task_args: [dict, None],
                 max_steps: int = pyrado.inf,
                 num_substeps: int = 1,
                 asset_file: str = 'mjcf/humanoid_20_5.xml',
                 asset_options: [gymapi.AssetOptions, None] = None,
                 ):
        """
        Constructor

        :param dt: time step size between physical interactions [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction, e.g `dict(state_des=np.zeros(42))`
        :param num_substeps: number of equal sub-intervals per interaction steps, e.g. `dt=1/100` and `num_substeps=4`
                             means that the agent interacts every 0.01s and the physics engine simulates every 0.0025s.
        :param asset_file: configuration file specifying the environment
        :param asset_options: import options affect the physical and visual characteristics of the model,
                              thus may have an impact on simulation stability and performance. See Isaac Gym doc.
        """
        Serializable._init(self, locals())
        super().__init__(dt, max_steps)

        # Create the carbongym-based implementation
        self.gym = gymapi.acquire_gym()
        self.impl = self.gym.create_sim(compute_device=0,  # GPU for the physics computation
                                        graphics_device=0,  # GPU for the rendering
                                        type=gymapi.SIM_FLEX)  # type can be SIM_PHYSX or SIM_FLEX

        # Change parameters of the overarching simulation
        # Basic parameters
        sim_params = self.gym.SimParams()
        sim_params.dt = dt
        sim_params.substeps = num_substeps
        # Flex-specific parameters
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
        sim_params.flex.shape_collision_margin = 0.01
        self.gym.set_sim_params(self.impl, sim_params)

        # Create workers
        self._num_envs = num_envs
        self.envs = [IsaacWorkerSimEnv(self.gym, dt, task_args, max_steps, num_substeps, asset_file, asset_options)
                     for i in range(num_envs)]

        # Initialize the domain parameters
        self._domain_param = dict()

        # Initialize spaces (identical for all sub-environments)
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._init_space = None
        self._create_spaces()

        # Initialize task
        self.task_args = task_args if task_args is not None else dict(state_des=None)
        self._task = self._create_task(self.task_args)

        # Initialize animation with Isaac Gym GUI
        self.viewer = None

    def __del__(self):
        """ Finalizer forwards to cleanup function. """
        self.cleanup()

    def cleanup(self):
        """ Destroy the viewer if there was one, and the physics simulator instance. """
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.impl)

    def _create_spaces(self):
        self._state_space = None
        self._obs_space = self._state_space.copy()
        self._init_space = SingularStateSpace(self.gym.get_env_rigid_body_states(self.envs[0], gymapi.STATE_ALL))
        self._act_space = None

    @classmethod
    def get_nominal_domain_param(cls):
        return {}  # raise NotImplementedError TODO implement in subclasses and throw error here

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

        if init_state.shape == self.obs_space.shape:
            # Allow setting the whole space
            if not self.obs_space.contains(init_state, verbose=True):
                pyrado.ValueErr(msg='The full init state must be within the state space!')
            self.state = init_state.copy()
        else:
            # Set the initial state determined by an element of the init space
            if not self.init_space.contains(init_state, verbose=True):
                pyrado.ValueErr(msg='The init state must be within init state space!')
            self.state = self._state_from_init(init_state)

        # Reset the task
        self._task.reset(env_spec=self.spec)

        # Return an observation
        return self.observe(self.state)

    def step(self, acts: np.ndarray) -> tuple:
        # apply the actions
        for env in self.envs:
            env.step(acts)

        # Compuate the next  Isaac Gym simulation
        self.gym.simulate(self.impl)
        self.gym.fetch_results(self.impl, True)

        # TODO do we get obs or the state?
        # self.state = self._state_from_obs(obs)  # only for the Python side

        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        info = dict(t=self._curr_step*self._dt)

        if done:
            # Add final reward if done
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)
        else:
            # Don't count the transition when done
            self._curr_step += 1

        return self.observe(self.state), self._curr_rew, done, info

    def render(self, mode: RenderMode = RenderMode(text=True), render_step: int = 1):
        if self._curr_step%render_step == 0:
            # Call base class
            super().render(mode)

            # Forward to Isaac Gym GUI
            if mode.video:
                if self.viewer is None:
                    self.viewer = self.gym.create_viewer(self.impl, self.gym.CameraProperties())

                # Update the viewer
                self.gym.step_graphics(self.impl)
                self.gym.draw_viewer(self.viewer, (self.impl, True))
                self.gym.sync_frame_time(self.impl)  # TODO move this to the rollout function?!










class IsaacWorkerSimEnv(SimEnv, Serializable):
    """
    Base class for simulated environments implemented in Nvidia's Isaac Gym

    .. note::
         Isaac Gym uses a y-up coordinate convention. They really have no mercy :)
    """

    def __init__(self,
                 gym,
                 dt: float,
                 task_args: [dict, None],
                 max_steps: int = pyrado.inf,
                 num_substeps: int = 1,
                 asset_file: str = 'mjcf/humanoid_20_5.xml',
                 asset_options: [gymapi.AssetOptions, None] = None,
                 ):
        """
        Constructor

        :param dt: time step size between physical interactions [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction, e.g `dict(state_des=np.zeros(42))`
        :param num_substeps: number of equal sub-intervals per interaction steps, e.g. `dt=1/100` and `num_substeps=4`
                             means that the agent interacts every 0.01s and the physics engine simulates every 0.0025s.
        :param asset_file: configuration file specifying the environment
        :param asset_options: import options affect the physical and visual characteristics of the model,
                              thus may have an impact on simulation stability and performance. See Isaac Gym doc.
        """
        Serializable._init(self, locals())
        super().__init__(dt, max_steps)

        # Create the carbongym-based implementation
        self._gym = gym
        self._impl = self._gym.create_sim(compute_device=0,  # GPU for the physics computation
                                          graphics_device=0,  # GPU for the rendering
                                          type=gymapi.SIM_FLEX)  # type can be SIM_PHYSX or SIM_FLEX

        # Change parameters of the overarching simulation
        # Basic parameters
        sim_params = self._gym.SimParams()
        sim_params.dt = dt
        sim_params.substeps = num_substeps
        # Flex-specific parameters
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
        sim_params.flex.shape_collision_margin = 0.01
        self._gym.set_sim_params(self._impl, sim_params)

        if False:
            # asset_options = gymapi.AssetOptions()
            # asset_options.fix_base_link = True
            # asset_options.armature = 0.01
            asset = self._gym.load_asset(sim=self._impl, rootpath=pyrado.ISAAC_ASSETS_DIR, filename=asset_file,
                                         options=asset_options)
        else:
            asset = self._gym.create_sphere(self._impl, 0.5)
        if asset is None:
            raise IOError("Failed to load asset")

        # Instantiate the actors (this needs to be done before working with the simulation)
        self._envs, self._actors = self._create_envs_and_actors(assets=[asset])

        # Initialize the domain parameters
        self._domain_param = dict()

        # Initialize spaces (identical for all sub-environments)
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._init_space = None
        self._create_spaces()

        # Initialize task
        self.task_args = task_args if task_args is not None else dict(state_des=None)
        self._task = self._create_task(self.task_args)

        # Initialize animation with Isaac Gym GUI
        self.viewer = None

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

    def _create_spaces(self):
        self._state_space = None
        self._obs_space = self._state_space.copy()
        self._init_space = SingularStateSpace(self._gym.get_env_rigid_body_states(self._envs[0], gymapi.STATE_ALL))
        self._act_space = None

    @property
    def task(self) -> Task:
        return self._task

    # @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        # Needs to implemented by subclasses
        return DesStateTask(self.spec, np.zeros(self.init_space.shape), MinusOnePerStepRewFcn())
        # raise NotImplementedError  TODO implement in subclasses and throw error here

    # @abstractmethod
    def _create_envs_and_actors(self, assets: Iterable) -> Tuple[list, list]:
        """
        Create Isaac Gym environments and actors. An actor is an instance of an asset.
        To add an actor to an environment, you must specify the source asset, desired pose, and a few other details.

        .. note::
            You cannot add or remove actors in an environment after you finish setting it up.

        :return: list of sub-environment handles and a list actor handles
        """
        # Define local extents and create parallel sub-environments (each has its own coordinate space)
        spacing = 2.
        lower = self._gym.Vec3(-spacing, 0., -spacing)
        upper = self._gym.Vec3(spacing, spacing, spacing)
        num_envs_per_row = 1

        envs, actors = [], []
        for i, asset in enumerate(assets):
            # Make space for an environment
            env_handle = self._gym.create_env(self._impl, lower, upper, num_envs_per_row)

            pose = self._gym.Transform()
            pose.p = self._gym.Vec3(0.0, 0.4, -1.0)  # local coordinates
            pose.r = self._gym.Quat(-0.707107, 0.0, 0.0, 0.707107)  # rotate from z-up to y-up coordinate system
            actor_handle = self._gym.create_actor(
                env_handle, asset, pose, name=f'actor_{i}',
                group=0,  # the actor will not collide with anything outside its collision group
                filter=0,  # bodies with the same filter bit will not collide
                segmentation=0
            )

            envs.append(env_handle)
            actors.append(actor_handle)

        print_cbt(f'Created {len(assets)} environments with actors.', 'w')
        return envs, actors

    @property
    def domain_param(self) -> dict:
        if not (self._envs and self._actors):
            raise pyrado.ValueErr(msg='Isaac Gym envs and actors must be created before calling domain_param!')

        for e, a in zip(self._envs, self._actors):
            shape_props = self._gym.get_actor_rigid_shape_properties(e, a)
            self._domain_param.update({f'{a.name}': shape_props.friction})

        return self._domain_param

    @domain_param.setter
    def domain_param(self, param: dict):
        if not isinstance(param, dict):
            raise pyrado.TypeErr(given=param, expected_type=dict)
        # Update the parameters
        self._domain_param.update(param)

        # Update spaces
        self._create_spaces()

        # Update task
        self._task = self._create_task(self.task_args)

    @classmethod
    def get_nominal_domain_param(cls):
        return {}  # raise NotImplementedError TODO implement in subclasses and throw error here

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

        if init_state.shape == self.obs_space.shape:
            # Allow setting the whole space
            if not self.obs_space.contains(init_state, verbose=True):
                pyrado.ValueErr(msg='The full init state must be within the state space!')
            self.state = init_state.copy()
        else:
            # Set the initial state determined by an element of the init space
            if not self.init_space.contains(init_state, verbose=True):
                pyrado.ValueErr(msg='The init state must be within init state space!')
            self.state = self._state_from_init(init_state)

        # Reset the task
        self._task.reset(env_spec=self.spec)

        # Return an observation
        return self.observe(self.state)

    def step(self, act: np.ndarray) -> tuple:
        # Current reward depending on the state (before step) and the (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self.task.step_rew(self.state, act, remaining_steps)

        # Apply actuator limits
        act = self._limit_act(act)

        force = act[self._envIndex]
        self._gym.apply_joint_effort(self._envPtr, self.sliderJoint, force)



        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        info = dict(t=self._curr_step*self._dt)

        if done:
            # Add final reward if done
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)
        else:
            # Don't count the transition when done
            self._curr_step += 1

        return self.observe(self.state), self._curr_rew, done, info
