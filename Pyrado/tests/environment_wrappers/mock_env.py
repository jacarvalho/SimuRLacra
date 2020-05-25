import numpy as np
import random

from pyrado.environments.pysim.base import SimEnv
from pyrado.utils.data_types import RenderMode


class MockEnv(SimEnv):
    """
    A mock version of SimEnv, used in env wrapper tests.
    Observation, action, and init spaces as well as the task are passed to the constructor.
    The step() method saves the last action into the last_act attribute. The action value is converted to a list to
    ease assertions. step() and reset() return next_obs as observation. If it is None, a random vector is returned.
    """

    def __init__(self, obs_space=None, act_space=None, init_space=None, task=None):
        # Set spaces
        self._obs_space = obs_space
        self._act_space = act_space
        self._init_space = init_space
        self._task = task

        # Set empty domain param
        self._domain_param = {}
        
        # Init check attributes
        self.next_obs = None
        self.next_reward = None
        self.next_step_done = False
        self.last_act = None
    
    @property
    def obs_space(self):
        if self._obs_space is None:
            raise NotImplementedError
        return self._obs_space
    
    @property
    def state_space(self):
        # Just use observation space here for now.
        if self._obs_space is None:
            raise NotImplementedError
        return self._obs_space

    @property
    def init_space(self):
        if self._init_space is None:
            raise NotImplementedError
        return self._init_space

    @property
    def act_space(self):
        if self._act_space is None:
            raise NotImplementedError
        return self._act_space

    @property
    def task(self):
        raise self._task

    def _create_task(self, state_des):
        pass  # unused

    @property
    def domain_param(self):
        return self._domain_param.copy()

    @domain_param.setter
    def domain_param(self, param):
        self._domain_param.clear()
        self._domain_param.update(param)

    def get_nominal_domain_param(self):
        return {}
    
    def _get_obs(self):
        # Return None if no obs space
        if self._obs_space is None:
            return None
        
        # Return random if no next_obs set
        if self.next_obs is None:
            return self._obs_space.sample_uniform()

        return np.array(self.next_obs)
    
    def reset(self, init_state=None, domain_param=None):
        # Init state is not needed for now.
        
        # Set domain params
        if domain_param is not None:
            self.domain_param = domain_param
        
        # Return next observation
        return self._get_obs()

    def step(self, act):
        # Store as last action as list, to simplify asserts
        if self._act_space is not None:
            self.last_act = list(act)
        
        # Return next observation
        obs = self._get_obs()
        
        # And next reward
        rew = self.next_reward
        if rew is None:
            rew = random.random()
            
        return obs, rew, self.next_step_done, dict()

    def render(self, mode=RenderMode(), render_step=1):
        # No visualization
        pass
