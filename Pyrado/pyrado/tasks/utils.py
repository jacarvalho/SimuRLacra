import numpy as np


def never_succeeded(err: np.ndarray = None) -> bool:
    """ The task is never marked successful, i.e. runs until failure or end of the episode. """
    return False


def proximity_succeeded(err: np.ndarray, thold_dist: float, dims: int = -1) -> bool:
    """
    The task is done successfully if the L2-norm of the selected dimensions of the state is smaller than some threshold.

    :param err: error in state, i.e. difference between desired and current state
    :param thold_dist: threshold for being solved
    :param dims: selected dimensions, i.e. indices, for the state, by default all dimensions are selected
    :return: `True` if successful
    """
    if dims == -1:
        # Select all dimensions for computing the distance
        l2_dist = np.linalg.norm(err, ord=2)
    else:
        # Select a subset of dimensions for computing the distance
        l2_dist = np.linalg.norm(err[dims], ord=2)
    return l2_dist < thold_dist
