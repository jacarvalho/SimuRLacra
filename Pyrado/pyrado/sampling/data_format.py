import numpy as np
import torch as to
from collections.abc import Sequence

import pyrado


def new_tuple(nt_type, values):
    """
    Create a new tuple of the same type as nt_type.
    This handles the constructor differences between tuple and NamedTuples

    :param nt_type: type of tuple
    :param values: values as sequence
    :return: new named tuple
    """
    if nt_type == tuple:
        # Use regular sequence-based ctor
        return tuple(values)
    else:
        # Use broadcast operator
        return nt_type(*values)


def to_format(data, data_format, data_type=None):
    """
    Convert the tensor data to the given data format.

    :param data: input data
    :param data_format: numpy or torch
    :param data_type: type to return data in. When None is passed, the data type is left unchanged.
    :return: numpy.ndarray or torch.Tensor
    """
    if data_format == 'numpy':
        if isinstance(data, np.ndarray):
            if data_type is None:
                return data
            else:
                return data.astype(data_type)
        elif isinstance(data, to.Tensor):
            if data_type is None:
                return data.cpu().detach().numpy()  # detach() is necessary, if the variable requires gradient
            else:
                return data.cpu().detach().numpy().astype(data_type)
        else:
            # Can't guarantee this, but let's try
            return np.asanyarray(data)

    elif data_format == 'torch':
        if isinstance(data, np.ndarray):
            try:
                if data_type is None:
                    return to.from_numpy(data)
                else:
                    return to.from_numpy(data).type(dtype=data_type)
            except ValueError:
                # Catch negative strides error
                data = np.array(data)
                if data_type is None:
                    return to.from_numpy(data)
                else:
                    return to.from_numpy(data).type(dtype=data_type)
        elif isinstance(data, to.Tensor):
            if data_type is None:
                return data
            else:
                return data.type(dtype=data_type)
        else:
            # Can't guarantee this, but let's try
            return to.tensor(data, dtype=to.get_default_dtype())
    else:
        raise pyrado.ValueErr(given=data_format, eq_constraint="'numpy' or 'torch'")


def stack_to_format(data: [dict, tuple, Sequence], data_format: str):
    """
    Stack the generic data in the given data format.
    For dicts, the dict elements are stacked individually. A list of dicts is treated as a dict of lists.

    :param data: input data
    :param data_format: 'numpy' or 'torch'
    :return: numpy `ndarray` or PyTorch `Tensor`, or `dict` of these
    """
    # Check recursion case
    if isinstance(data, dict):
        # Stack entries of dict
        keys = data.keys()
        return {k: stack_to_format(data[k], data_format) for k in keys}
    if isinstance(data, tuple):
        # Stack entries of tuple
        return new_tuple(type(data), (stack_to_format(part, data_format) for part in data))

    if isinstance(data, Sequence):
        if isinstance(data[0], dict):
            # Stack dict entries separately
            keys = data[0].keys()
            return {k: stack_to_format([d[k] for d in data], data_format) for k in keys}
        if isinstance(data[0], tuple):
            # Stack tuple entries separately
            tsize = len(data[0])
            return new_tuple(type(data[0]), (stack_to_format([d[i] for d in data], data_format) for i in range(tsize)))

        # Convert elements, then stack
        if data_format == 'numpy':
            return np.stack([to_format(d, data_format) for d in data])
        elif data_format == 'torch':
            if not isinstance(data[0], (to.Tensor, np.ndarray)):
                # Torch's stack doesn't work with non-tensors, so this is more efficient
                return to.tensor(data, dtype=to.get_default_dtype())
            return to.stack([to_format(d, data_format) for d in data])
        else:
            raise pyrado.ValueErr(given=data_format, eq_constraint="'numpy' or 'torch'")

    else:
        return to_format(data, data_format)


def cat_to_format(data: [dict, tuple, Sequence], data_format: str):
    """
    Concatenate the generic data in the given data format.
    For dicts, the dict elements are stacked individually. A list of dicts is treated as a dict of lists.

    :param data: input data
    :param data_format: numpy or torch
    :return: numpy.ndarray or torch.Tensor, or dict of these
    """
    # Check recursion case
    if isinstance(data, dict):
        # Stack entries of dict
        keys = data.keys()
        return {k: cat_to_format(data[k], data_format) for k in keys}
    if isinstance(data, tuple):
        # Stack entries of tuple
        return new_tuple(type(data), (cat_to_format(part, data_format) for part in data))

    if isinstance(data, Sequence):
        if isinstance(data[0], dict):
            # Stack dict entries separately
            keys = data[0].keys()
            return {k: cat_to_format([d[k] for d in data], data_format) for k in keys}
        if isinstance(data[0], tuple):
            # Stack tuple entries separately
            tsize = len(data[0])
            return new_tuple(type(data[0]), (cat_to_format([d[i] for d in data], data_format) for i in range(tsize)))

        # Convert elements, then stack
        if data_format == 'numpy':
            return np.concatenate([to_format(d, data_format) for d in data])
        elif data_format == 'torch':
            if not isinstance(data[0], (to.Tensor, np.ndarray)):
                # Torch's stack doesn't work with non-tensors, so this is more efficient
                return to.tensor(data, dtype=to.get_default_dtype())
            return to.cat([to_format(d, data_format) for d in data])
        else:
            raise pyrado.ValueErr(given=data_format, eq_constraint="'numpy' or 'torch'")

    else:
        return to_format(data, data_format)
