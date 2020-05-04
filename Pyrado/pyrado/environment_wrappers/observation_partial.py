import numpy as np

import pyrado
from pyrado.environments.base import Env
from pyrado.environment_wrappers.base import EnvWrapperObs
from pyrado.spaces.base import Space
from init_args_serializer.serializable import Serializable


class ObsPartialWrapper(EnvWrapperObs, Serializable):
    """ Environment wrapper which creates a partial observation by masking certain elements """

    def __init__(self, wrapped_env: Env, mask: list = None, indices: list = None, keep_selected: bool = False):
        """
        Constructor

        :param wrapped_env: environment to wrap
        :param mask: mask array, entries with 1 are dropped (behavior can be inverted by keep_selected=True)
        :param indices: indices to drop, ignored if mask is specified.
                        If the observation space is labeled, labels can be used as indices.
        :param keep_selected: set to true to keep the mask entries with 1/the specified indices and drop the others.
        """
        Serializable._init(self, locals())

        super(ObsPartialWrapper, self).__init__(wrapped_env)

        # Parse selection
        if mask is not None:
            mask = np.array(mask, dtype=np.bool)
            if not mask.shape == wrapped_env.obs_space.shape:
                raise pyrado.ShapeErr(given=mask, expected_match=wrapped_env.obs_space)
        else:
            # Parse indices
            assert indices is not None, 'Either mask or indices must be specified'
            mask = wrapped_env.obs_space.create_mask(wrapped_env.obs_space, indices)
        # Invert if needed
        if keep_selected:
            self.keep_mask = mask
        else:
            self.keep_mask = np.logical_not(mask)

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.keep_mask]

    def _process_obs_space(self, space: Space) -> Space:
        return space.subspace(self.keep_mask)
