import numpy as np
from copy import deepcopy
from typing import Sequence

from pyrado.spaces.base import Space
from pyrado.utils.input_output import print_cbt


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
        raise NotImplementedError

    def subspace(self, idcs: [int, slice]):
        # Subspace of this CompoundSpace and not of the individual spaces
        return self._spaces[idcs]

    def shrink(self, new_lo: np.ndarray, new_up: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def cat(spaces: [list, tuple]):
        raise NotImplementedError

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        valid = any([s.contains(cand) for s in self._spaces])
        if not valid and verbose:
            print_cbt(f'Violated all of the {len(self._spaces)} subspaces!', 'r')
            for s in self._spaces:
                s.contains(cand, verbose)
        return valid

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        # Sample a subspace and then sample from this subspace
        idx = np.random.randint(len(self._spaces))
        return self._spaces[idx].sample_uniform()
