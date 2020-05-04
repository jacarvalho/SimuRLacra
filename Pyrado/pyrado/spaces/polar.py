import numpy as np
from typing import Sequence

from pyrado.spaces import BoxSpace


class Polar2DPosSpace(BoxSpace):
    """
    Samples positions on a 2-dim torus, i.e. the area between two concentric circles.
    Can also be a section of a 2-dim torus, i.e. not a full circle.
    """

    def __init__(self,
                 bound_lo: [float, list, np.ndarray],
                 bound_up: [float, list, np.ndarray],
                 labels: Sequence[str] = None):
        """
        Constructor

        :param bound_lo: minimal distance and the minimal angle (polar coordinates)
        :param bound_up: maximal distance and the maximal angle (polar coordinates)
        :param labels: label for each dimension of the space
        """
        assert bound_lo.size == bound_up.size == 2
        # Actually, this space is a BoxSpace
        super().__init__(bound_lo, bound_up, labels=labels)

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        # Get a random sample from the polar space
        sample = super().sample_uniform()
        # Transform the positions to the cartesian space
        return np.array([sample[0] * np.cos(sample[1]), sample[0] * np.sin(sample[1])])

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        assert cand.size == 2
        # Transform candidate to polar space
        x, y = cand[0], cand[1]
        polar = np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x)])  # arctan2 returns in range [-pi, pi] -> check bounds
        # Query base
        return super().contains(polar, verbose=verbose)


class Polar2DPosVelSpace(BoxSpace):
    """
    Samples positions on a 2-dim torus, i.e. the area between 2 circles augmented with cartesian velocities.
    Can also be a section of a 2-dim torus, i.e. not a full circle.
    """

    def __init__(self,
                 bound_lo: [float, list, np.ndarray],
                 bound_up: [float, list, np.ndarray],
                 labels: Sequence[str] = None):
        """
        Constructor

        :param bound_lo: minimal distance, the minimal angle, and the 2D minimal cartesian initial velocity
        :param bound_up: maximal distance, the maximal angle, and the 2D minimal cartesian initial velocity
        :param labels: label for each dimension of the space
        """
        assert bound_lo.size == bound_up.size == 4
        # Actually, this space is a BoxSpace
        super().__init__(bound_lo, bound_up, labels=labels)

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        # Get a random sample from the half-polar / half-cartesian space
        sample = super().sample_uniform()
        # Transform the positions to the cartesian space
        sample[:2] = np.array([sample[0] * np.cos(sample[1]), sample[0] * np.sin(sample[1])])
        return sample

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        assert cand.size == 4
        # Transform candidate to polar space
        x, y = cand[0], cand[1]
        polar = np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x)])  # arctan2 returns in range [-pi, pi] -> check bounds
        # Query base
        return super().contains(np.r_[polar, cand[2:]], verbose=verbose)