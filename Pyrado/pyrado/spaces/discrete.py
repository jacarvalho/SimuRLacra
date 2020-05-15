import numpy as np
from tabulate import tabulate
from typing import Sequence

import pyrado
from pyrado.spaces.base import Space
from pyrado.utils.input_output import color_validity


class DiscreteSpace(Space):
    """ Discrete space implemented as a ndarray containing all possible integer-valued elements (unsorted) """

    def __init__(self, eles: [np.ndarray, list], labels: Sequence[str] = None):
        """
        Constructor

        :param eles: 2D ndarray of all elements listed along the columns
        :param labels: label element of the space. This is useful for giving the states and actions names to later
                       identify them (e.g. for plotting).
        """
        if isinstance(eles, np.ndarray):
            # Make sure the dimension of the state is along the first array dimension
            self.eles = eles if eles.ndim == 2 else eles.reshape(-1, 1)
        elif isinstance(eles, list):
            self.eles = np.array(eles, dtype=np.int)
            # Make sure the dimension of the state is along the first array dimension
            self.eles = eles if self.eles.ndim == 2 else self.eles.reshape(-1, 1)
        else:
            raise pyrado.TypeErr(given=eles, expected_type=[np.ndarray, list])

        self.eles = np.atleast_2d(self.eles)
        self.bound_lo = np.min(self.eles, axis=0)
        self.bound_up = np.max(self.eles, axis=0)

        # Process the labels
        if labels is not None:
            labels = np.array(labels, dtype=object)
            if not labels.shape == self.shape:
                raise pyrado.ShapeErr(given=labels, expected_match=self)
            self._labels = labels
        else:
            self._labels = np.empty(self.shape, dtype=object)
            self._labels.fill(None)

    def __str__(self):
        """ Get an information string. """
        return f'DiscreteSpace id: {id(self)}\nelements: {self.eles}'

    @property
    def shape(self) -> tuple:
        return self.bound_lo.shape  # equivalent to bound_up.shape

    @property
    def flat_dim(self) -> int:
        return self.eles.shape[0]

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def _members(self) -> tuple:
        # Compare elements, the upper and lower bound are derived and not enough.
        return self.eles, self.labels

    def shrink(self, new_lo: np.ndarray, new_up: np.ndarray):
        raise NotImplementedError

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        # Check the candidate
        if not cand.shape == self.shape:
            raise pyrado.ShapeErr(given=cand, expected_match=self)
        if np.isnan(cand).any():
            raise pyrado.ValueErr(
                msg=f'At least one value is NaN!' +
                    tabulate([list(self.labels), [*color_validity(cand, np.invert(np.isnan(cand)))]], headers='firstrow')
            )

        # Cast and approximately compare
        return np.any(np.isclose(self.eles, cand.astype(self.eles.dtype)))

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        idx = np.random.randint(self.flat_dim, size=1)
        # Return randomly selected column
        return self.eles[idx, :].flatten()

    def project_to(self, ele: np.ndarray) -> np.ndarray:
        if not self.contains(ele):
            # Return the element from the space closest to the given element
            absdiffs = np.abs(ele - self.eles)
            idx_closest = np.argmin(absdiffs)
            return self.eles[idx_closest, :]
        else:
            return ele

    @staticmethod
    def cat(spaces: [list, tuple]):
        """
        Concatenate multiple instances of `DiscreteSpace`.

        .. note::
            This function does not check if the dimensions of the BoxSpaces are correct!

        :param spaces: list or tuple of spaces
        """
        # Remove None elements for convenience
        spaces = [s for s in spaces if s is not None]

        eles_cat, labels_cat = [], []
        for s in spaces:
            if not isinstance(s, DiscreteSpace):
                raise pyrado.TypeErr(given=s, expected_type=DiscreteSpace)
            eles_cat.extend(s.eles)
            labels_cat.extend(s.labels)

        # Merge list of arrays (with potentially unequal size) to one array
        if all(isinstance(e, np.ndarray) for e in eles_cat):
            eles_cat = np.concatenate(eles_cat, axis=0)

        return DiscreteSpace(eles_cat, labels=None)  # omitting labels for now
