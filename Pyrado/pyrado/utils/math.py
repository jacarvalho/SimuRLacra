import numpy as np
import torch as to

import pyrado
from pyrado.utils.normalizing import normalize


class UnitCubeProjector:
    """ Project to a unit qube $[0, 1]^d$ and back using explicit bounds """

    def __init__(self, bound_lo: [np.ndarray, to.Tensor], bound_up: [np.ndarray, to.Tensor]):
        """
        Constructor

        :param bound_lo: lower bound
        :param bound_up: upper bound
        """
        self.bound_lo = bound_lo
        self.bound_up = bound_up

    def project_to(self, x: [np.ndarray, to.Tensor]) -> [np.ndarray, to.Tensor]:
        """
        Normalize every dimension individually using the stored explicit bounds and the L_1 norm.

        :param x: input ndarray or Tensor
        :return: element of the unit cube
        """
        span = self.bound_up - self.bound_lo
        span[span == 0] = 1.  # avoid division by 0
        if isinstance(x, to.Tensor):
            return (x - self.bound_lo)/span
        elif isinstance(x, np.ndarray):
            return (x - self.bound_lo.numpy())/span.numpy()
        else:
            raise pyrado.TypeErr(given=x, expected_type=[np.ndarray, to.Tensor])

    def project_back(self, x: [np.ndarray, to.Tensor]) -> [np.ndarray, to.Tensor]:
        """
        Revert the previous normalization using the stored explicit bounds

        :param x: input ndarray or Tensor
        """
        span = self.bound_up - self.bound_lo
        if isinstance(x, to.Tensor):
            return span*x + self.bound_lo
        elif isinstance(x, np.ndarray):
            return span.numpy()*x + self.bound_lo.numpy()
        else:
            raise pyrado.TypeErr(given=x, expected_type=[np.ndarray, to.Tensor])


def cov(x: to.Tensor, data_along_rows: bool = False):
    """
    Compute the covariance matrix given data.

    .. note::
        Only real valued matrices are supported

    :param x: matrix containing multiple observations of multiple variables
    :param data_along_rows: if `True` the variables are stacked along the columns, else they are along the rows
    :return: covariance matrix given the data
    """
    if x.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if x.dim() < 2:
        x = x.view(1, -1)
    if data_along_rows and x.size(0) != 1:
        # Transpose if necessary
        x = x.t()

    num_samples = x.size(1)
    if num_samples < 2:
        raise pyrado.ShapeErr(msg='Need at least 2 samples to compute the covariance!')

    x -= to.mean(x, dim=1, keepdim=True)
    return x.matmul(x.t()).squeeze()/(num_samples - 1)


def explained_var(y_mdl: [np.ndarray, to.Tensor], y_obs: [np.ndarray, to.Tensor]) -> (np.ndarray, to.Tensor):
    """
    Calculate proportion of the variance "explained" by the model (see coefficient of determination R^2)
    .. note:: R^2 = 0.49 implies that 49% of the variability of the dependent variable has been accounted for.

    :param y_mdl: predictions from the model
    :param y_obs: observed data (ground truth)
    :return: proportion of the explained variance (i.e. R^2 value)
    """
    if isinstance(y_mdl, np.ndarray) and isinstance(y_obs, np.ndarray):
        ss_total = np.sum(np.power(y_obs - np.mean(y_obs), 2))  # proportional to the variance of the data
        ss_res = np.sum(np.power(y_obs - y_mdl, 2))  # proportional to the residual variance
        return 1. - ss_res/ss_total  # R^2

    elif isinstance(y_mdl, to.Tensor) and isinstance(y_obs, to.Tensor):
        ss_total = to.sum(to.pow(y_obs - to.mean(y_obs), 2))
        ss_res = to.sum(to.pow(y_obs - y_mdl, 2))
        return to.tensor(1.) - ss_res/ss_total

    else:
        raise pyrado.TypeErr(given=y_mdl, expected_type=[np.ndarray, to.Tensor])


def logmeanexp(x: to.Tensor, dim: int = 0) -> to.Tensor:
    """
    Numerically stable way to compute $\log( 1/N \sum_{i=1}^N \exp(x) )$

    :param x: input tensor
    :param dim: dimension to torch.logsumexp along
    :return: $\log( 1/N \sum_{i=1}^N \exp(x) )$
    """
    return to.logsumexp(x, dim=dim) - to.log(to.tensor(x.shape[dim], dtype=to.get_default_dtype()))


def cosine_similarity(x: to.Tensor, y: to.Tensor) -> to.Tensor:
    r"""
    Compute the cosine similarity between two tensors $D_cos(x,y) = \frac{x^T y}{|x| \cdot |y|}$.

    :param x: input tensor
    :param y: input tensor
    :return: cosine similarity value
    """
    if not isinstance(x, to.Tensor):
        raise pyrado.TypeErr(given=x, expected_type=to.Tensor)
    if not isinstance(y, to.Tensor):
        raise pyrado.TypeErr(given=y, expected_type=to.Tensor)

    x_normed = normalize(x, order=2)
    y_normed = x_normed if y is x else normalize(y, order=2)
    return x_normed.dot(y_normed)


def clamp(inp: to.Tensor, lo: to.Tensor, up: to.Tensor) -> to.Tensor:
    """
    Clip the entries of a tensor by the entries of a lower bound and an upper bound tensor.

    :param inp: input tensor
    :param lo: lower bound for each entry
    :param lo: upper bound for each entry
    :return: clipped tensor
    """
    if not (lo < up).all():
        raise pyrado.ValueErr(msg='The lower bounds needs to be element-wise smaller than the upper bound.')
    return to.max(to.min(inp, up), lo)


def clamp_symm(inp: to.Tensor, up_lo: to.Tensor) -> to.Tensor:
    """
    Symmetrically clip the entries of a tensor by the entries of a tensor with only positive entries.
    One use case of this function is if you want to clip the parameter update by a ratio of the old parameters.

    :param inp: input tensor
    :param up_lo: upper and (negative of the) lower bound for each entry
    :return: clipped tensor
    """
    if not (up_lo > 0).all():
        raise pyrado.ValueErr(given=up_lo, g_constraint='0')
    return to.max(to.min(inp.clone(), up_lo), -up_lo)
