import torch as to
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

import pyrado


def sample_from_hyper_sphere_surface(num_dim: int, method: str) -> to.Tensor:
    """
    Sampling from the surface of a multidimensional unit sphere.

    .. seealso::
        [1] G. Marsaglia, "Choosing a Point from the Surface of a Sphere", Ann. Math. Statist., 1972

    :param num_dim: number of dimensions of the sphere
    :param method: approach used to acquire the samples
    :return: sample with L2-norm equal 1
    """
    assert num_dim > 0
    num_dim = int(num_dim)

    if method == 'uniform':
        # Initialization
        ones = to.ones((num_dim,))
        udistr = Uniform(low=-ones, high=ones)
        sum_squares = pyrado.inf

        # Sample candidates until criterion is met
        while sum_squares >= 1:
            sample = udistr.sample()
            sum_squares = sample.dot(sample)

        # Return scaled sample
        return sample / to.sqrt(sum_squares)

    elif method == 'normal':
        # Sample fom standardized normal
        sample = Normal(loc=to.zeros((num_dim,)), scale=to.ones((num_dim,))).sample()

        # Return scaled sample
        return sample / to.norm(sample, p=2)

    elif method == 'Marsaglia':
        if not (num_dim == 3 or num_dim == 4):
            raise pyrado.ValueErr(msg="Method 'Marsaglia' is only defined for 3-dim space")
        else:
            # Initialization
            ones = to.ones((2,))
            udistr = Uniform(low=-ones, high=ones)
            sum_squares = pyrado.inf

            # Sample candidates until criterion is met
            while sum_squares >= 1:
                sample = udistr.sample()
                sum_squares = sample.dot(sample)

            if num_dim == 3:
                # Return scaled sample
                return to.tensor([2 * sample[0] * to.sqrt(1 - sum_squares),
                                  2 * sample[1] * to.sqrt(1 - sum_squares),
                                  1 - 2 * sum_squares])
            else:
                # num_dim = 4
                sum_squares2 = pyrado.inf
                while sum_squares2 >= 1:
                    sample2 = udistr.sample()
                    sum_squares2 = sample2.dot(sample2)
                # Return scaled sample
                return to.tensor([sample[0], sample[1],
                                  sample2[0] * to.sqrt((1 - sum_squares) / sum_squares2),
                                  sample2[1] * to.sqrt((1 - sum_squares) / sum_squares2)])
    else:
        raise pyrado.ValueErr(given=method, eq_constraint="'uniform', 'normal', or 'Marsaglia'")
