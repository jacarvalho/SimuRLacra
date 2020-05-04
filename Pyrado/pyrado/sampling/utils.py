import random
from itertools import islice


def gen_batches(batch_size, data_size):
    """
    Helper function for doing SGD on mini-batches.

    :param batch_size: number of samples in each mini-batch
    :param data_size: total number of samples
    :return: generator for lists of random indices of sub-samples

    Example:
        If num_rollouts = 2 and data_size = 5, then the output might be
        out = ((0, 3), (2, 1), (4,)).
    """
    idx_all = random.sample(range(data_size), data_size)
    idx_iter = iter(idx_all)
    return iter(lambda: list(islice(idx_iter, batch_size)), [])


def gen_ordered_batches(batch_size, data_size):
    """
    Helper function for doing SGD on mini-batches.

    :param batch_size: number of samples in each mini-batch
    :param data_size: total number of samples
    :return: generator for lists of random indices of sub-samples

    Example:
        If num_rollouts = 2 and data_size = 5, then the output will be
        out = ((2, 3), (0, 1), (4,)).
    """
    from math import ceil
    num_batches = int(ceil(data_size / batch_size))

    # Create a list of lists, each containing num_rollouts ordered elements
    idcs_all = list(range(data_size))
    idcs_batches = [idcs_all[i * batch_size:i * batch_size + batch_size] for i in range(num_batches)]

    # Yield a random sample from the list of lists
    idcs_batches_rand = random.sample(idcs_batches, len(idcs_batches))
    idx_iter = iter(idcs_batches_rand)
    return iter(islice(idx_iter, num_batches))
