import numpy as np
import pytest
import torch as to

from pyrado.utils.tensor import stack_tensor_list, stack_tensor_dict_list, insert_tensor_col, \
    atleast_2D, atleast_3D



@pytest.mark.pytorch_augmentation
@pytest.mark.parametrize(
    'x', [
        to.tensor(3.),
        to.rand(1, ),
        to.rand(2, ),
        to.rand(1, 2),
        to.rand(2, 1),
        to.rand(2, 3),
        to.rand(2, 3, 4)
    ], ids=['sclar', 'scalar_1D', 'vec_1D', 'vec_2D', 'vec_2D_T', 'arr_2D', 'arr_3D'])
def test_atleast_2D(x):
    x_al2d = atleast_2D(x)
    assert x_al2d.ndim >= 2

    # We want to mimic the numpy function
    x_np = np.atleast_2d(x.numpy())
    assert np.all(x_al2d.numpy() == x_np)



@pytest.mark.pytorch_augmentation
@pytest.mark.parametrize(
    'x', [
        to.tensor(3.),
        to.rand(1, ),
        to.rand(2, ),
        to.rand(1, 2),
        to.rand(2, 1),
        to.rand(2, 3),
        to.rand(2, 3, 4)
    ], ids=['sclar', 'scalar_1D', 'vec_1D', 'vec_2D', 'vec_2D_T', 'arr_2D', 'arr_3D'])
def test_atleast_3D(x):
    x_al3d = atleast_3D(x)
    assert x_al3d.ndim >= 2

    # We want to mimic the numpy function
    x_np = np.atleast_3d(x.numpy())
    assert np.all(x_al3d.numpy() == x_np)



@pytest.mark.pytorch_augmentation
def test_stack_tensors():
    tensors = [
        to.tensor([1, 2, 3]),
        to.tensor([2, 3, 4]),
        to.tensor([4, 5, 6]),
    ]

    stack = stack_tensor_list(tensors)

    to.testing.assert_allclose(stack, to.tensor([
        [1, 2, 3],
        [2, 3, 4],
        [4, 5, 6],
    ]))



@pytest.mark.pytorch_augmentation
def test_stack_tensors_scalar():
    tensors = [1, 2, 3]
    stack = stack_tensor_list(tensors)
    to.testing.assert_allclose(stack, to.tensor([1, 2, 3]))



@pytest.mark.pytorch_augmentation
def test_stack_tensor_dicts():
    tensors = [
        {'multi': [1, 2], 'single': 1},
        {'multi': [3, 4], 'single': 2},
        {'multi': [5, 6], 'single': 3},
    ]
    stack = stack_tensor_dict_list(tensors)
    to.testing.assert_allclose(stack['single'], to.tensor([1, 2, 3]))
    to.testing.assert_allclose(stack['multi'], to.tensor([[1, 2], [3, 4], [5, 6]]))



@pytest.mark.pytorch_augmentation
@pytest.mark.parametrize(
    'orig, col', [
        (to.rand((1, 1)), to.zeros(1, 1)),
        (to.rand((3, 3)), to.zeros(3, 1)),
    ], ids=['1_dim', '3_dim']
)
def test_insert_tensor_col(orig, col):
    for col_idx in range(orig.shape[1] + 1):  # also check appending case
        result = insert_tensor_col(orig, col_idx, col)
        # Check number of rows and columns
        assert orig.shape[0] == result.shape[0]
        assert orig.shape[1] == result.shape[1] - 1
        # Check the values
        to.testing.assert_allclose(result[:, col_idx], col.squeeze())
