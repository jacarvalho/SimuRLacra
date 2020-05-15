import numpy as np
from tabulate import tabulate
from typing import Sequence

import pyrado
from pyrado.spaces.base import Space
from pyrado.utils.input_output import color_validity


class BoxSpace(Space):
    """ Multidimensional box space. (This class can also be used to describe a sphere.) """

    def __init__(self,
                 bound_lo: [float, list, np.ndarray],
                 bound_up: [float, list, np.ndarray],
                 shape: [tuple, int] = None,
                 labels: Sequence[str] = None):
        """
        Constructor

        :param bound_lo: array_like containing the minimal values for each dimension of the space
        :param bound_up: array_like containing the maximal values for each dimension of the space
        :param shape: tuple specifying the shape, usefull if all lower and upper bounds are identical
        :param labels: label for each dimension of the space (e.g. list of strings)
        """
        if shape is not None:
            # The bounds are scalars
            self.bound_lo = np.ones(shape)*bound_lo
            self.bound_up = np.ones(shape)*bound_up
        else:
            # Cast the bounds into arrays if necessary
            try:
                self.bound_lo = np.atleast_1d(np.array(bound_lo))
            except TypeError:
                raise pyrado.TypeErr(given=bound_lo, expected_type=[float, list, np.ndarray])
            try:
                self.bound_up = np.atleast_1d(np.array(bound_up))
            except TypeError:
                raise pyrado.TypeErr(given=bound_up, expected_type=[float, list, np.ndarray])

            if self.bound_lo.shape != self.bound_up.shape:
                raise pyrado.ShapeErr(given=bound_lo, expected_match=bound_up)

        # Process the labels
        if labels is not None:
            labels = np.array(labels, dtype=object)
            if not labels.shape == self.shape:
                raise pyrado.ShapeErr(given=labels, expected_match=self)
            self._labels = labels
        else:
            self._labels = np.empty(self.shape, dtype=object)
            self._labels.fill(None)

    # This nice idea comes from https://stackoverflow.com/questions/45164691/recommended-way-to-implement-eq-and-hash
    def _members(self):
        # Return members relevant for equals. Hash isn't supported by numpy arrays, so we don't support it too.
        return self.bound_lo, self.bound_up, self._labels

    @property
    def shape(self) -> tuple:
        return self.bound_lo.shape  # equivalent to bound_up.shape

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def subspace(self, idcs: Sequence[int]):
        if not isinstance(idcs, np.ndarray) or idcs.dtype != np.dtype(np.bool_):
            # Interpret as index list
            mask = self.create_mask(idcs)
        else:
            mask = idcs

        labels = None
        if self.labels is not None:
            labels = self.labels[mask]

        if len(self.shape) == 1:
            bound_lo = np.atleast_1d(self.bound_lo[mask])
            bound_up = np.atleast_1d(self.bound_up[mask])
        elif len(self.shape) == 2 and self.shape[1] == 1:
            # We assume only box spaces with one dimension, i.e. no images
            bound_lo = np.atleast_1d(self.bound_lo[mask]).reshape(-1, 1)
            bound_up = np.atleast_1d(self.bound_up[mask]).reshape(-1, 1)
            labels = labels.reshape(-1, 1)
        else:
            raise NotImplementedError

        return BoxSpace(bound_lo, bound_up, labels=labels)

    def shrink(self, new_lo: np.ndarray, new_up: np.ndarray):
        if not isinstance(new_lo, np.ndarray):
            raise pyrado.TypeErr(given=new_lo, expected_type=np.ndarray)
        if not isinstance(new_up, np.ndarray):
            raise pyrado.TypeErr(given=new_up, expected_type=np.ndarray)
        if not new_lo.shape == new_up.shape:
            raise pyrado.ShapeErr(given=new_up, expected_match=new_lo)
        if not (new_lo >= self.bound_lo).all():
            raise pyrado.ValueErr(msg='At least one new lower bound is too low!')
        if not (new_up <= self.bound_up).all():
            raise pyrado.ValueErr(msg='At least one new upper bound is too high!')

        shrinked_box = self.copy()
        shrinked_box.bound_lo = new_lo
        shrinked_box.bound_up = new_up
        return shrinked_box

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        # Check the candidate validity (shape and NaN values)
        if not cand.shape == self.shape:
            raise pyrado.ShapeErr(given=cand, expected_match=self)
        if np.isnan(cand).any():
            raise pyrado.ValueErr(
                msg=f'At least one value is NaN!' +
                    tabulate([list(self.labels), [*color_validity(cand, np.invert(np.isnan(cand)))]], headers='firstrow')
            )

        # Check upper and lower bound separately
        check_lo = (cand >= self.bound_lo).astype(int)
        check_up = (cand <= self.bound_up).astype(int)
        idcs_valid = np.bitwise_and(check_lo, check_up)

        if np.all(idcs_valid):
            return True
        else:
            if verbose:
                print(tabulate([
                    ['lower bound ', *color_validity(self.bound_lo, check_lo)],
                    ['candidate ', *color_validity(cand, idcs_valid)],
                    ['upper bound ', *color_validity(self.bound_up, check_up)]
                ], headers=[""] + list(self.labels)))
            return False

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        bl = self.bound_lo.copy()
        bu = self.bound_up.copy()
        # replace inf bounds to be able to work with the RNG
        bl[bl == -np.inf] = -concrete_inf
        bu[bu == np.inf] = concrete_inf

        return np.random.uniform(bl, bu)

    def project_to(self, ele: np.ndarray) -> np.ndarray:
        if not self.contains(ele):
            return np.clip(ele, self.bound_lo, self.bound_up)
        else:
            return ele

    @staticmethod
    def cat(spaces: [list, tuple]):
        """
        Concatenate BoxSpaces.

        :param spaces: list or tuple of spaces
        .. note:: This function does not check if the dimensions of the BoxSpaces are correct!
        """
        # Remove None elements for convenience
        spaces = [s for s in spaces if s is not None]

        bound_lo_cat, bound_up_cat, labels_cat = [], [], []
        for s in spaces:
            if not isinstance(s, BoxSpace):
                raise pyrado.TypeErr(given=s, expected_type=BoxSpace)
            bound_lo_cat.extend(s.bounds[0])
            bound_up_cat.extend(s.bounds[1])
            labels_cat.extend(s.labels)

        return BoxSpace(bound_lo_cat, bound_up_cat, labels=labels_cat)
