"""
Only tested for 1-dim inputs, e.g. time series of rewards.
"""
import numpy as np
import torch as to

import pyrado


def standardize(data: [np.ndarray, to.Tensor], eps: float = 1e-8) -> [np.ndarray, to.Tensor]:
    r"""
    Standardize the input data to make it $~ N(0, 1)$.

    :param data: input ndarray or Tensor
    :param eps: factor for numerical stability
    :return: standardized ndarray or Tensor
    """
    if isinstance(data, np.ndarray):
        return (data - np.mean(data))/(np.std(data) + float(eps))
    elif isinstance(data, to.Tensor):
        return (data - to.mean(data))/(to.std(data) + float(eps))
    else:
        raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])


class Standardizer:
    """ A stateful standardizer that remembers the mean and standard deviation for later un-standardization """

    def __init__(self):
        self.mean = None
        self.std = None

    def standardize(self, data: [np.ndarray, to.Tensor], eps: float = 1e-8) -> [np.ndarray, to.Tensor]:
        """
        Standardize the input data to make it $~ N(0, 1)$ and store the input's mean and standard deviation.

        :param data: input ndarray or Tensor
        :param eps: factor for numerical stability
        :return: standardized ndarray or Tensor
        """
        if isinstance(data, np.ndarray):
            self.mean = np.mean(data)
            self.std = np.std(data)
            return (data - self.mean)/(self.std + float(eps))
        elif isinstance(data, to.Tensor):
            self.mean = to.mean(data)
            self.std = to.std(data)
            return (data - self.mean)/(self.std + float(eps))
        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

    def unstandardize(self, data: [np.ndarray, to.Tensor]) -> [np.ndarray, to.Tensor]:
        """
        Revert the previous standardization of the input data to make it $~ N(\mu, \sigma)$.

        :param data: input ndarray or Tensor
        :return: un-standardized ndarray or Tensor
        """
        if self.mean is None or self.std is None:
            raise pyrado.ValueErr(msg='Use standardize before unstandardize!')

        # Input type must match stored type
        if isinstance(data, np.ndarray) and isinstance(self.mean, np.ndarray):
            pass
        elif isinstance(data, to.Tensor) and isinstance(self.mean, to.Tensor):
            pass
        elif isinstance(data, np.ndarray) and isinstance(self.mean, to.Tensor):
            self.mean = self.mean.numpy()
            self.std = self.std.numpy()
        elif isinstance(data, to.Tensor) and isinstance(self.mean, np.ndarray):
            self.mean = to.from_numpy(self.mean)
            self.std = to.from_numpy(self.std)

        x_unstd = data*self.std + self.mean
        return x_unstd


class RunningStandardizer:
    """ Implementation of Welford's online algorithm """

    def __init__(self):
        """ Constructor """
        self._mean = None
        self._sum_sq_diffs = None  # a.k.a M2
        self._iter = 0

    def reset(self):
        """ Reset internal variables. """
        self._mean = None
        self._sum_sq_diffs = None
        self._iter = 0

    def __repr__(self):
        return f'RunningStandardizer ID: {id(self)}\n' \
               f'mean: {self._mean}\nss_diffs: {self._sum_sq_diffs}\niter: {self._iter}'

    def __call__(self, data: [np.ndarray, to.Tensor], axis: int = 0):
        """
        Update the internal variables and standardize the input.

        :param data: input data to be standardized
        :param axis: axis to standardized along
        :return: standardized data
        """
        if isinstance(data, np.ndarray):
            # Process element wise (keeps dim) or average along one axis
            mean = np.mean(data, axis=axis)

            self._iter += 1

            # Handle first iteration separately
            if self._iter <= 1:
                # delta = x
                self._mean = mean
                self._sum_sq_diffs = mean * mean
                return data
            else:
                # Update mean
                delta_prev = mean - self._mean
                self._mean += delta_prev / self._iter

                # Update sum of squares of differences from the current mean
                delta_curr = mean - self._mean
                self._sum_sq_diffs += delta_prev * delta_curr

                # Calculate the unbiased sample variance
                var_sample = self._sum_sq_diffs / (self._iter - 1)

                # Return normalized data
                return (data - self._mean) / np.sqrt(var_sample)

        elif isinstance(data, to.Tensor):
            # Process element wise (keeps dim) or average along one axis
            mean = to.mean(data, dim=axis).to(to.get_default_dtype())

            self._iter += 1

            # Handle first iteration separately
            if self._iter <= 1:
                # delta = x
                self._mean = mean
                self._sum_sq_diffs = mean * mean
                return data
            else:
                # Update mean
                delta_prev = mean - self._mean
                self._mean += delta_prev / self._iter

                # Update sum of squares of differences from the current mean
                delta_curr = mean - self._mean
                self._sum_sq_diffs += delta_prev * delta_curr

                # Calculate the unbiased sample variance
                var_sample = self._sum_sq_diffs / (self._iter - 1)

                # Return normalized data
                return (data - self._mean) / to.sqrt(var_sample)

        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])
