import numpy as np
from abc import ABC
from tabulate import tabulate
from typing import Sequence

import pyrado
from pyrado.spaces.base import Space
from pyrado.utils.input_output import color_validity


class EmptySpace(Space, ABC):
    """ A space with no content """

    def _members(self) -> tuple:
        # We're a singleton, compare by id
        return id(self),

    @property
    def bounds(self) -> tuple:
        return np.array([]), np.array([])

    @property
    def labels(self) -> (np.ndarray, None):
        return np.array([], dtype=np.object)

    @property
    def shape(self) -> tuple:
        return ()

    def shrink(self, new_lo: np.ndarray, new_up: np.ndarray):
        raise NotImplementedError("Cannot shrink empty space!")

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        # Check the candidate
        if not cand.shape == self.shape:
            raise pyrado.ShapeErr(given=cand, expected_match=self)
        if np.isnan(cand).any():
            raise pyrado.ValueErr(
                msg=f'At least one value is NaN!' +
                    tabulate([list(self.labels), [*color_validity(cand, np.invert(np.isnan(cand)))]], headers='firstrow')
            )
        return True

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        return np.array([])

    def project_to(self, ele: np.ndarray) -> np.ndarray:
        return np.array([])

    @staticmethod
    def cat(spaces: [list, tuple]):
        if not all(isinstance(s, EmptySpace) for s in spaces):
            raise pyrado.TypeErr(given=spaces, expected_type=Sequence[EmptySpace])
        return EmptySpace
