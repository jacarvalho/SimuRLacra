import torch as to

import pyrado


def atleast_2D(x: to.Tensor) -> to.Tensor:
    """
    Mimic the numpy function.

    :param x: any tensor
    :return: an at least 2-dim tensor
    """
    if x.ndim >= 2:
        return x
    elif x.ndim == 1:
        # First dim is the batch size (1 in this case)
        return x.unsqueeze(0)
    else:
        return x.view(1, 1)


def atleast_3D(x: to.Tensor) -> to.Tensor:
    """
    Mimic the numpy function.

    :param x: any tensor
    :return: an at least 3-dim tensor
    """
    if x.ndim >= 3:
        return x
    elif x.ndim == 2:
        # First dim is the batch size (1 in this case). We add dimensions at the end.
        return x.unsqueeze(-1)
    elif x.ndim == 1:
        if x.size() == to.Size([1]):
            # First dim is the batch size (1 in this case).
            # We add dimensions at the end, but could also do it at the beginning.
            return x.unsqueeze(-1).unsqueeze(-1)
        else:
            # First dim is the batch size (1 in this case). We add one dim at the end and at the beginning.
            return x.unsqueeze(-1).unsqueeze(0)
    else:
        return x.view(1, 1, 1)


def stack_tensor_list(tensor_list: list) -> to.Tensor:
    """
    Covenience funtion for stacking a list of tensors

    :param tensor_list: list of tensors to stack (along a new dim)
    :return: tensor of at least 1-dim
    """
    if not to.is_tensor(tensor_list[0]):
        # List of scalars (probably)
        return to.tensor(tensor_list)
    return to.stack(tensor_list)


def stack_tensor_dict_list(tensor_dict_list: list) -> dict:
    """
    Stack a list of dictionaries of {tensors or dict of tensors}.

    :param tensor_dict_list: a list of dicts of {tensors or dict of tensors}.
    :return: a dict of {stacked tensors or dict of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def insert_tensor_col(x: to.Tensor, idx: int, col: to.Tensor) -> to.Tensor:
    """
    Insert a column into a PyTorch Tensor.

    :param x: original tensor
    :param idx: column index where to insert the column
    :param col: tensor to insert
    :return: tensor with new column at index idx
    """
    assert isinstance(x, to.Tensor)
    assert isinstance(idx, int) and -1 <= idx <= x.shape[1]
    assert isinstance(col, to.Tensor)
    if not x.shape[0] == col.shape[0]:
        raise pyrado.ShapeErr(
            msg=f'Number of rows does not match! Original: {x.shape[0]}, column to insert: {col.shape[0]}')

    # Concatenate along columns
    if 0 <= idx < x.shape[1]:
        return to.cat((x[:, :idx], col, x[:, idx:]), dim=1)
    else:
        # idx == -1 or idx == shape[1]
        return to.cat((x, col), dim=1)
