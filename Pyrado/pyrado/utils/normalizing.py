import numpy as np
import torch as to

import pyrado


class RunningNormalizer:
    """ Normalizes given data based on the history of observed data, such that all outputs are in range [-1, 1] """

    def __init__(self):
        """ Constructor """
        self._bound_lo = None
        self._bound_up = None
        self.eps = 1e-3
        self._iter = 0

    def reset(self):
        """ Reset internal variables. """
        self._bound_lo = None
        self._bound_up = None
        self._iter = 0

    def __repr__(self):
        return f'RunningNormalizer ID: {id(self)}\n' \
            f'bound_lo: {self._bound_lo}\nbound_up: {self._bound_up}\niter: {self._iter}'

    def __call__(self, data: [np.ndarray, to.Tensor]):
        """
        Update the internal variables and normalize the input.

        :param data: input data to be standardized
        :return: normalized data in [-1, 1]
        """
        if isinstance(data, np.ndarray):
            data_2d = np.atleast_2d(data)
            data_min = np.min(data_2d, axis=0)
            data_max = np.max(data_2d, axis=0)
            self._iter += 1

            # Handle first iteration separately
            if self._iter <= 1:
                self._bound_lo = data_min
                self._bound_up = data_max
            else:
                if not self._bound_lo.shape == data_min.shape:
                    raise pyrado.ShapeErr(given=data_min, expected_match=self._bound_lo)

                # Update bounds with element wise
                self._bound_lo = np.fmin(self._bound_lo, data_min)
                self._bound_up = np.fmax(self._bound_up, data_max)

            # Make sure that the bounds do not collapse (e.g. for one sample)
            if np.linalg.norm(self._bound_up - self._bound_lo, ord=1) < self.eps:
                self._bound_lo -= self.eps / 2
                self._bound_up += self.eps / 2

        elif isinstance(data, to.Tensor):
            data_2d = data.view(-1, 1) if data.ndim < 2 else data
            data_min, _ = to.min(data_2d, dim=0)
            data_max, _ = to.max(data_2d, dim=0)
            self._iter += 1

            # Handle first iteration separately
            if self._iter <= 1:
                self._bound_lo = data_min
                self._bound_up = data_max
            else:
                if not self._bound_lo.shape == data_min.shape:
                    raise pyrado.ShapeErr(given=data_min, expected_match=self._bound_lo)

                # Update bounds with element wise
                self._bound_lo = to.min(self._bound_lo, data_min)
                self._bound_up = to.max(self._bound_up, data_max)

            # Make sure that the bounds do not collapse (e.g. for one sample)
            if to.norm(self._bound_up - self._bound_lo, p=1) < self.eps:
                self._bound_lo -= self.eps / 2
                self._bound_up += self.eps / 2

        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

        # Return standardized data
        return (data - self._bound_lo) / (self._bound_up - self._bound_lo) * 2 - 1


def normalize(x: [np.ndarray, to.Tensor], axis: int = -1, order: int = 1) -> (np.ndarray, to.Tensor):
    """
    Normalize a numpy `ndarray` or a PyTroch `Tensor` without changing the input.
    Choosing `axis=1` and `norm_order=1` makes all columns of sum to 1.

    :param x: input to normalize
    :param axis: axis of the array to normalize along
    :param order: order of the norm (e.g., L_1 norm: absolute values, L_2 norm: quadratic values)
    :return: normalized array
    """
    if isinstance(x, np.ndarray):
        norm_x = np.atleast_1d(np.linalg.norm(x, ord=order, axis=axis))  # calculate norm over axis
        norm_x[norm_x == 0] = 1.  # avoid division by 0
        return x/np.expand_dims(norm_x, axis)  # element wise division
    elif isinstance(x, to.Tensor):
        norm_x = to.norm(x, p=order, dim=axis)  # calculate norm over axis
        norm_x[norm_x == 0] = 1.
        return x/norm_x.unsqueeze(axis)  # element wise division
    else:
        raise pyrado.TypeErr(given=x, expected_type=[np.array, to.Tensor])