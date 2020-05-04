import gym.envs
import gym.spaces as gs
import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.base import Space
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.utils.data_types import RenderMode


def _space_to_ps(gym_space) -> [BoxSpace, DiscreteSpace]:
    """
    Convert a space from OpenAIGym to Pyrado.

    :param gym_space: space object from OpenAIGym
    :return: space object in Pyrado
    """
    if isinstance(gym_space, gs.Box):
        return BoxSpace(gym_space.low, gym_space.high)
    if isinstance(gym_space, gs.Discrete):
        return DiscreteSpace(range(gym_space.n))
    else:
        raise pyrado.TypeErr(msg=f'Unsupported space form gym {gym_space}')


class GymEnv(SimEnv, Serializable):
    """ A Wrapper to use the classical control environments of OpenAI Gym like Pyrado environments """

    name: str = 'gym-cc'

    def __init__(self, env_name: str):
        """
        Constructor

        .. note::
            Pyrado only supports the classical control environments from OpenAI Gym.
            See https://github.com/openai/gym/tree/master/gym/envs/classic_control

        :param env_name: name of the OpenAI Gym environment, e.g. 'MountainCar-v0', 'CartPole-v1', 'Acrobot-v1',
                         'MountainCarContinuous-v0','Pendulum-v0'
        """
        Serializable._init(self, locals())

        # Initialize basic variables
        if env_name == 'MountainCar-v0':
            dt = 0.02  # there is no dt in the source file
        elif env_name == 'CartPole-v1':
            dt = 0.02
        elif env_name == 'Acrobot-v1':
            dt = 0.2
        elif env_name == 'MountainCarContinuous-v0':
            dt = 0.02  # there is no dt in the source file
        elif env_name == 'Pendulum-v0':
            dt = 0.05
        else:
            raise NotImplementedError(f'GymEnv does not wrap the environment {env_name}.')
        super().__init__(dt)

        # Create the gym environment
        self._gym_env = gym.envs.make(env_name)

        # Create spaces compatible to Pyrado
        self._obs_space = _space_to_ps(self._gym_env.observation_space)
        self._act_space = _space_to_ps(self._gym_env.action_space)

    @property
    def state_space(self) -> Space:
        # Copy of obs_space since the OpenAI gym has no dedicated state space
        return self._obs_space

    @property
    def obs_space(self) -> Space:
        return self._obs_space

    @property
    def init_space(self) -> None:
        # OpenAI Gym environments do not have an init_space
        return None

    @property
    def act_space(self) -> Space:
        return self._act_space

    def _create_task(self, state_des: [np.ndarray, None]) -> None:
        return None

    @property
    def task(self):
        # Doesn't have any
        return None

    @property
    def domain_param(self):
        # Doesn't have any
        return {}

    @domain_param.setter
    def domain_param(self, param):
        # Ignore
        pass

    @classmethod
    def get_nominal_domain_param(cls):
        # Doesn't have any
        return {}

    def reset(self, init_state=None, domain_param=None):
        return self._gym_env.reset()

    def step(self, act) -> tuple:
        return self._gym_env.step(act)

    def render(self, mode: RenderMode = RenderMode(), render_step: int = 1):
        if mode.video:
            return self._gym_env.render()

    def close(self):
        return self._gym_env.close()
