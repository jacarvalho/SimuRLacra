import numpy as np
from init_args_serializer import Serializable
from typing import Mapping, Optional

import pyrado
from pyrado.environment_wrappers.base import EnvWrapperObs
from pyrado.environments.base import Env
from pyrado.spaces.box import BoxSpace
from pyrado.utils.normalizing import RunningNormalizer


class ObsNormWrapper(EnvWrapperObs, Serializable):
    """
    Environment wrapper which normalizes the observation space using the bounds from the environment or
    hard-coded bounds, such that all values are in range [-1, 1]
    """

    def __init__(self,
                 wrapped_env: Env,
                 explicit_lb: Mapping[str, float] = None,
                 explicit_ub: Mapping[str, float] = None):
        """
        Constructor

        :param wrapped_env: environment to wrap
        :param explicit_lb: dict to override the environment's lower bound; by default (`None`) this is ignored;
                            the keys are space labels, the values the new bound for that labeled entry
        :param explicit_ub: dict to override the environment's upper bound; by default (`None`) this is ignored;
                            the keys are space labels, the values the new bound for that labeled entry
        """
        Serializable._init(self, locals())
        super().__init__(wrapped_env)

        # Explicitly override the bounds if desired
        self.explicit_lb = explicit_lb
        self.explicit_ub = explicit_ub

        # Check that the

    @staticmethod
    def override_bounds(bounds: np.ndarray,
                        override: Optional[Mapping[str, float]],
                        bound_label: str,
                        names: np.ndarray) -> np.ndarray:
        """
        Override a given bound. This function is useful if some entries of the observation space have an infinite bound
        and/or you want to specify a certain bound

        :param bounds: bound to override
        :param override: value to override with
        :param bound_label: label of the bound to override
        :param names: e.g. lower or upper
        :return: new bound created from a copy of the old bound
        """
        if not override:
            return bounds
        # Override in copy of bounds
        bc = bounds.copy()
        for idx, name in np.ndenumerate(names):
            ov = override.get(name)
            if ov is not None:
                # Apply override
                bc[idx] = ov
            elif np.isinf(bc[idx]):
                # Report unbounded entry
                raise pyrado.ValueErr(msg=f'{name} entry of {bound_label} bound is infinite and not overridden.'
                                          f'Cannot apply normalization.')
            else:
                # Do nothing if ov is None
                pass
        return bc

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        # Get the bounds of the inner observation space
        wos = self.wrapped_env.obs_space
        lb, ub = wos.bounds

        # Override the bounds if desired
        lb = ObsNormWrapper.override_bounds(lb, self.explicit_lb, 'lower', wos.labels)
        ub = ObsNormWrapper.override_bounds(ub, self.explicit_ub, 'upper', wos.labels)

        # Normalize observation
        obs_norm = (obs - lb)/(ub - lb)*2 - 1
        return obs_norm

    def _process_obs_space(self, space: BoxSpace) -> BoxSpace:
        if not isinstance(space, BoxSpace):
            raise NotImplementedError('Only implemented ObsNormWrapper._process_obs_space() for BoxSpace!')
        # Get the bounds of the inner observation space
        lb, ub = space.bounds

        # Override the bounds if desired
        lb_ov = ObsNormWrapper.override_bounds(lb, self.explicit_lb, 'lower', space.labels)
        ub_ov = ObsNormWrapper.override_bounds(ub, self.explicit_ub, 'upper', space.labels)

        if any(lb_ov == -pyrado.inf):
            raise pyrado.ValueErr(msg=f'At least one element of the lower bounds is (negative) infinite:\n'
                                      f'(overwritten) bound: {lb_ov}\nnames: {space.labels}')
        if any(ub_ov == pyrado.inf):
            raise pyrado.ValueErr(msg=f'At least one element of the upper bound is (positive) infinite:\n'
                                      f'(overwritten) bound: {ub_ov}\nnames: {space.labels}')

        # Report actual bounds, which are not +-1 for overridden fields
        lb_norm = (lb - lb_ov)/(ub_ov - lb_ov)*2 - 1
        ub_norm = (ub - lb_ov)/(ub_ov - lb_ov)*2 - 1
        return BoxSpace(lb_norm, ub_norm, labels=space.labels)


class ObsRunningNormWrapper(EnvWrapperObs, Serializable):
    """
    Environment wrapper which normalizes the observation space using the bounds from the environment or
    hard-coded bounds, such that all values are in range [-1, 1]
    """

    def __init__(self, wrapped_env: Env):
        """
        Constructor

        :param wrapped_env: environment to wrap
        """
        Serializable._init(self, locals())
        super().__init__(wrapped_env)

        # Explicitly override the bounds if desired
        self.normalizer = RunningNormalizer()

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.normalizer(obs)

    def _process_obs_space(self, space: BoxSpace) -> BoxSpace:
        if not isinstance(space, BoxSpace):
            raise NotImplementedError('Only implemented ObsRunningNormWrapper._process_obs_space() for BoxSpace!')

        # Return space with same shape but bounds from -1 to 1
        return BoxSpace(-np.ones(space.shape), np.ones(space.shape), labels=space.labels)
