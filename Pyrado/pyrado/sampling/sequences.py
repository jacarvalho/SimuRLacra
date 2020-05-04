import numpy as np


def sequence_const(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0

    :param x_init: constant values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    assert isinstance(x_init, (int, float, np.ndarray))
    assert isinstance(iter, int) and iter >= 0
    assert dtype == int or dtype == float

    if isinstance(x_init, np.ndarray):
        dim = len(x_init)
    else:
        dim = 1

    if iter > 0:
        x_seq = np.zeros((iter + 1, dim))
        x_seq[0, :] = x_init
        for i in range(1, iter + 1):
            # Fill all entries with the same value
            x_seq[i, :] = x_init
        # Return and type casting
        if dim == 1:
            # If x is not a np.ndarray, the last element of the sequence should also not be a np.ndarray
            return dtype(x_seq[-1, :]), x_seq.astype(dtype)
        elif dim > 1:
            return x_seq[-1, :].astype(dtype), x_seq.astype(dtype)

    elif iter == 0:
        if isinstance(x_init, np.ndarray):
            return x_init.copy().T, x_init.copy().T
        else:
            return x_init, x_init


def sequence_plus_one(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0 + n

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    assert isinstance(x_init, (int, float, np.ndarray))
    assert isinstance(iter, int) and iter >= 0
    assert dtype == int or dtype == float

    if isinstance(x_init, np.ndarray):
        dim = len(x_init)
    else:
        dim = 1

    if iter > 0:
        x_seq = np.zeros((iter + 1, dim))
        x_seq[0, :] = x_init
        for i in range(1, iter + 1):
            # non-exponential growth
            x_seq[i, :] = x_seq[0, :] + i
        # Return and type casting
        if dim == 1:
            # If x is not a np.ndarray, the last element of the sequence should also not be a np.ndarray
            return dtype(x_seq[-1, :]), x_seq.astype(dtype)
        elif dim > 1:
            return x_seq[-1, :].astype(dtype), x_seq.astype(dtype)

    elif iter == 0:
        if isinstance(x_init, np.ndarray):
            return x_init.copy().T, x_init.copy().T
        else:
            return x_init, x_init


def sequence_add_init(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0 * n

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    assert isinstance(x_init, (int, float, np.ndarray))
    assert isinstance(iter, int) and iter >= 0
    assert dtype == int or dtype == float

    if isinstance(x_init, np.ndarray):
        dim = len(x_init)
    else:
        dim = 1

    if iter > 0:
        x_seq = np.zeros((iter + 1, dim))
        x_seq[0, :] = x_init
        for i in range(1, iter + 1):
            # non-exponential growth
            x_seq[i, :] = x_seq[0, :] * (i + 1)
        # Return and type casting
        if dim == 1:
            # If x is not a np.ndarray, the last element of the sequence should also not be a np.ndarray
            return dtype(x_seq[-1, :]), x_seq.astype(dtype)
        elif dim > 1:
            return x_seq[-1, :].astype(dtype), x_seq.astype(dtype)

    elif iter == 0:
        if isinstance(x_init, np.ndarray):
            return x_init.copy().T, x_init.copy().T
        else:
            return x_init, x_init


def sequence_rec_double(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_{n-1} * 2

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    assert isinstance(x_init, (int, float, np.ndarray))
    assert isinstance(iter, int) and iter >= 0
    assert dtype == int or dtype == float

    if isinstance(x_init, np.ndarray):
        dim = len(x_init)
    else:
        dim = 1

    if iter > 0:
        x_seq = np.zeros((iter + 1, dim))
        x_seq[0, :] = x_init
        for i in range(1, iter + 1):
            # exponential growth
            x_seq[i, :] = x_seq[i - 1, :] * 2.
        # Return and type casting
        if dim == 1:
            # If x is not a np.ndarray, the last element of the sequence should also not be a np.ndarray
            return dtype(x_seq[-1, :]), x_seq.astype(dtype)
        elif dim > 1:
            return x_seq[-1, :].astype(dtype), x_seq.astype(dtype)

    elif iter == 0:
        if isinstance(x_init, np.ndarray):
            return x_init.copy().T, x_init.copy().T
        else:
            return x_init, x_init


def sequence_sqrt(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0 * sqrt(n)

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    assert isinstance(x_init, (int, float, np.ndarray))
    assert isinstance(iter, int) and iter >= 0
    assert dtype == int or dtype == float

    if isinstance(x_init, np.ndarray):
        dim = len(x_init)
    else:
        dim = 1

    if iter > 0:
        x_seq = np.zeros((iter + 1, dim))
        x_seq[0, :] = x_init
        for i in range(1, iter + 1):
            # non-exponential growth
            x_seq[i, :] = x_seq[0, :] * np.sqrt(i + 1)  # i+1 because sqrt(1) = 1
        # Return and type casting
        if dim == 1:
            # If x is not a np.ndarray, the last element of the sequence should also not be a np.ndarray
            return dtype(x_seq[-1, :]), x_seq.astype(dtype)
        elif dim > 1:
            return x_seq[-1, :].astype(dtype), x_seq.astype(dtype)

    elif iter == 0:
        if isinstance(x_init, np.ndarray):
            return x_init.copy().T, x_init.copy().T
        else:
            return x_init, x_init


def sequence_rec_sqrt(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_{n-1} * sqrt(n)

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    assert isinstance(x_init, (int, float, np.ndarray))
    assert isinstance(iter, int) and iter >= 0
    assert dtype == int or dtype == float

    if isinstance(x_init, np.ndarray):
        dim = len(x_init)
    else:
        dim = 1

    if iter > 0:
        x_seq = np.zeros((iter + 1, dim))
        x_seq[0, :] = x_init
        for i in range(1, iter + 1):
            # exponential growth
            x_seq[i, :] = x_seq[i - 1, :] * np.sqrt(i + 1)  # i+1 because sqrt(1) = 1
        # Return and type casting
        if dim == 1:
            # If x is not a np.ndarray, the last element of the sequence should also not be a np.ndarray
            return dtype(x_seq[-1, :]), x_seq.astype(dtype)
        elif dim > 1:
            return x_seq[-1, :].astype(dtype), x_seq.astype(dtype)

    elif iter == 0:
        if isinstance(x_init, np.ndarray):
            return x_init.copy().T, x_init.copy().T
        else:
            return x_init, x_init


def sequence_nlog2(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0 * n * log2(n+2), with log2 being the base 2 logarithm

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    assert isinstance(x_init, (int, float, np.ndarray))
    assert isinstance(iter, int) and iter >= 0
    assert dtype == int or dtype == float

    if isinstance(x_init, np.ndarray):
        dim = len(x_init)
    else:
        dim = 1

    if iter > 0:
        x_seq = np.zeros((iter + 1, dim))
        x_seq[0, :] = x_init
        for i in range(1, iter + 1):
            # non-exponential growth
            x_seq[i, :] = x_seq[0, :] * i * np.log2(i + 2)  # i+2 because log2(1) = 0 and log2(2) = 1
        # Return and type casting
        if dim == 1:
            # If x is not a np.ndarray, the last element of the sequence should also not be a np.ndarray
            return dtype(x_seq[-1, :]), x_seq.astype(dtype)
        elif dim > 1:
            return x_seq[-1, :].astype(dtype), x_seq.astype(dtype)

    elif iter == 0:
        if isinstance(x_init, np.ndarray):
            return x_init.copy().T, x_init.copy().T
        else:
            return x_init, x_init
