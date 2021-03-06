from abc import abstractmethod
from init_args_serializer import Serializable
import numpy as np

import pyrado
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode


class EnvWrapper(Env, Serializable):
    """ Base for all environment wrappers. Delegates all environment methods to the wrapped environment. """

    def __init__(self, wrapped_env: Env):
        """
        Constructor

        :param wrapped_env: environment to wrap
        """
        if not isinstance(wrapped_env, Env):
            raise pyrado.TypeErr(given=wrapped_env, expected_type=Env)

        Serializable._init(self, locals())

        self._wrapped_env = wrapped_env

    @property
    def name(self) -> str:
        """ Get the wrapped environment's abbreviated name. """
        return self._wrapped_env.name

    @property
    def wrapped_env(self) -> [Env, SimEnv]:
        """ Get the wrapped environment of this wrapper. """
        return self._wrapped_env

    @property
    def state_space(self) -> Space:
        return self._wrapped_env.state_space

    @property
    def obs_space(self) -> Space:
        return self._wrapped_env.obs_space

    @property
    def act_space(self) -> Space:
        return self._wrapped_env.act_space

    @property
    def init_space(self) -> Space:
        if isinstance(self._wrapped_env, (SimEnv, EnvWrapper)):
            return self._wrapped_env.init_space
        else:
            raise NotImplementedError

    @property
    def dt(self):
        return self._wrapped_env.dt

    @dt.setter
    def dt(self, dt: float):
        self._wrapped_env.dt = dt

    @property
    def curr_step(self) -> int:
        return self._wrapped_env.curr_step

    @property
    def max_steps(self) -> int:
        return self._wrapped_env.max_steps

    @max_steps.setter
    def max_steps(self, num_steps: int):
        self._wrapped_env.max_steps = num_steps

    def _create_task(self, task_args: dict) -> Task:
        return self._wrapped_env._create_task(task_args)

    @property
    def task(self) -> Task:
        return self._wrapped_env.task

    @property
    def domain_param(self) -> dict:
        """
        These are the environment's domain parameters, which are synonymous to the parameters used by the simulator to
        run the physics simulation (e.g., masses, extents, or friction coefficients). The property domain_param includes
        all parameters that can be perturbed a.k.a. randomized, but there might also be additional parameters.
        """
        param = self._wrapped_env.domain_param
        self._save_domain_param(param)
        return param

    @domain_param.setter
    def domain_param(self, param: dict):
        """
        Set the domain parameters. The changes are only applied at the next call of the reset function.
        """
        self._load_domain_param(param)
        self._wrapped_env.domain_param = param

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        """
        Reset the environment to its initial state and optionally set different domain parameters.

        :param init_state: set explicit initial state if not None
        :param domain_param: set explicit domain parameters if not None
        :return obs: initial observation of the state.
        """
        if domain_param is not None:
            self._load_domain_param(domain_param)
        return self._wrapped_env.reset(init_state, domain_param)

    def step(self, act: np.ndarray) -> tuple:
        """
        Perform one time step of the simulation. When a terminal condition is met, the reset function is called.

        :param act: action to be taken in the step
        :return tuple of obs, reward, done, and info:
                obs : current observation of the environment
                reward: reward depending on the selected reward function
                done: indicates whether the episode has ended
                env_info: contains diagnostic information about the environment
        """
        return self._wrapped_env.step(act)

    def render(self, mode: RenderMode, render_step: int = 1):
        self._wrapped_env.render(mode, render_step)

    def close(self):
        return self._wrapped_env.close()

    def _load_domain_param(self, param: dict):
        """ Called by the domain_param setter. Use to load wrapper-specific params. Does nothing by default. """
        pass

    def _save_domain_param(self, param: dict):
        """ Called by the domain_param getter. Use to store wrapper-specific params. Does nothing by default. """
        pass


class EnvWrapperAct(EnvWrapper):
    """
    Base class for environment wrappers modifying the action.
    Override _process_action to pass a modified action vector to the wrapped environment.
    If necessary, you should also override _process_action_space to report the correct one.
    """

    @abstractmethod
    def _process_act(self, act: np.ndarray):
        """
        Return the modified action vector to be passed to the wrapped environment.

        :param act: action vector (should not be modified in place)
        :return: changed action vector
        """
        raise NotImplementedError

    def _process_act_space(self, space: Space):
        """
        Return the modified action space. Override if the operation defined in _process_action affects
        shape or bounds of the action vector.
        :param space: inner env action space
        :return: action space to report for this env
        """
        return space

    def step(self, act: np.ndarray) -> tuple:
        # Modify action
        mod_act = self._process_act(act)

        # Delegate to base/wrapped
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperObs
        return super().step(mod_act)

    @property
    def act_space(self) -> Space:
        # Process space
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperObs
        return self._process_act_space(super().act_space)


class EnvWrapperObs(EnvWrapper):
    """
    Base class for environment wrappers modifying the observation.
    Override _process_obs to pass a modified observation vector to the wrapped environment.
    If necessary, you should also override _process_obs_space to report the correct one.
    """

    @abstractmethod
    def _process_obs(self, obs: np.ndarray):
        """
        Return the modified observation vector to be returned from this environment.

        :param obs: observation from the inner environment
        :return: changed observation vector
        """
        raise NotImplementedError

    def _process_obs_space(self, space: Space) -> Space:
        """
        Return the modified observation space.
        Override if the operation defined in _process_obs affects shape or bounds of the observation vector.
        :param space: inner env observation space
        :return: action space to report for this env
        """
        return space

    @property
    def obs_space(self) -> Space:
        # Process space
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        return self._process_obs_space(super().obs_space)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset inner environment
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        init_obs = super().reset(init_state, domain_param)

        # Return processed observation
        return self._process_obs(init_obs)

    def step(self, act: np.ndarray) -> tuple:
        # Step inner environment
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        obs, rew, done, info = super().step(act)

        # Return processed observation
        return self._process_obs(obs), rew, done, info
