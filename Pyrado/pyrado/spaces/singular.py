import numpy as np
from typing import Sequence

from pyrado.spaces import BoxSpace


class SingularStateSpace(BoxSpace):
    """ Space which always returns the same initial state (trivial space) """

    def __init__(self, fixed_state: np.ndarray, labels: Sequence[str] = None):
        """
        Constructor

        :param fixed_state: the initial state
        :param labels: label for each dimension of the space
        """
        # Call BoxSpace constructor
        super().__init__(fixed_state, fixed_state, labels=labels)
        self._fixed_state = fixed_state

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        return self._fixed_state.copy()