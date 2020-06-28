import torch as to
import torch.nn as nn
from torch import optim

if __name__ == '__main__':
    params = nn.Parameter(to.tensor([2., 4.]), requires_grad=False)
    finite_diffs = to.tensor([0.1, -0.1]).repeat(5, 1)  # for testing purposes, every loop we access the next row
    optimizer = optim.SGD([params], lr=1, momentum=0, nesterov=False, dampening=0)

    for i in range(finite_diffs.shape[0]):
        print(f'before: {params}')
        optimizer.zero_grad()  # also clears the reference
        params.grad = finite_diffs[i, :].detach()  # zero_grad() can not zero one row of finite_diffs, so we detach
        optimizer.step()
        print(f'after: {params}')
        print(f'{finite_diffs}')
