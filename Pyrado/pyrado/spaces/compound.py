import numpy as np
from copy import deepcopy
from typing import Sequence

from pyrado.spaces.base import Space


class CompoundSpace(Space):
    """ Space consisting of other spaces """

    def __init__(self, spaces: Sequence[Space]):
        """
        Constructor

        :param spaces: list or tuple of spaces to sample from randomly
        """
        self._spaces = deepcopy(spaces)

    @property
    def shape(self):
        # Return all shapes
        return (s for s in self._spaces)

    def _members(self):
        # Return the subspaces
        return self._spaces

    def project_to(self, ele: np.ndarray):
        return NotImplementedError

    def subspace(self, idcs: [int, slice]):
        return self._spaces[idcs]

    def shrink(self, new_lo: np.ndarray, new_up: np.ndarray):
        return NotImplementedError

    @staticmethod
    def cat(spaces: [list, tuple]):
        raise NotImplementedError

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        return any([s.contains(cand, verbose) for s in self._spaces])

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        # Sample a subspace and then sample from this subspace
        idx = np.random.randint(len(self._spaces))
        return self._spaces[idx].sample_uniform()
