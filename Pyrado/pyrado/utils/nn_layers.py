from math import ceil, sqrt
from typing import Callable

import torch as to
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single

import pyrado


class ScaleLayer(nn.Module):
    """ Layer which scales the output of the input using a learnable scaling factor """

    def __init__(self, in_features: int, init_weight: float = 1.):
        """
        Constructor

        :param in_features: size of each input sample
        :param init_weight: initial scaling factor
        """
        super().__init__()
        self.weight = nn.Parameter(init_weight*to.ones(in_features, dtype=to.get_default_dtype()), requires_grad=True)

    def forward(self, inp: to.Tensor) -> to.Tensor:
        # Element-wise product
        return inp*self.weight


class PositiveScaleLayer(nn.Module):
    """ Layer which scales (strictly positive) the input using a learnable scaling factor """

    def __init__(self, in_features: int, init_weight: float = 1.):
        """
        Constructor

        :param in_features: size of each input sample
        :param init_weight: initial scaling factor
        """
        if not init_weight > 0:
            raise pyrado.ValueErr(given=init_weight, g_constraint='0')

        super().__init__()
        self.log_weight = nn.Parameter(to.log(init_weight*to.ones(in_features, dtype=to.get_default_dtype())),
                                       requires_grad=True)

    def forward(self, inp: to.Tensor) -> to.Tensor:
        # Element-wise product
        return inp*to.exp(self.log_weight)


class IndiNonlinLayer(nn.Module):
    """
    Layer subtracts a bias from the input, multiplies the result with a strictly positive sacaling factor, and then
    applies the provided nonlinearity. The scaling and the bias are learnable parameters.
    """

    def __init__(self,
                 in_features: int,
                 nonlin: Callable,
                 bias: bool,
                 weight: bool = True,
                 init_weight: float = 1.,
                 init_bias: float = 0.):
        """
        Constructor

        :param in_features: size of each input sample
        :param nonlin: nonlinearity
        :param bias: if `True`, a learnable bias is subtracted, else no bias is used
        :param weight: if `True` (default), the input is multiplied with a learnable scaling factor
        :param init_weight: initial scaling factor
        :param init_bias: initial bias
        """
        if not init_weight > 0:
            raise pyrado.ValueErr(given=init_weight, g_constraint='0')

        super().__init__()
        self._nonlin = nonlin
        if weight:
            self.log_weight = nn.Parameter(to.log(init_weight*to.ones(in_features, dtype=to.get_default_dtype())),
                                           requires_grad=True)
        else:
            self.log_weight = None
        if bias:
            self.bias = nn.Parameter(init_bias*to.ones(in_features, dtype=to.get_default_dtype()), requires_grad=True)
        else:
            self.bias = None

    def forward(self, inp: to.Tensor) -> to.Tensor:
        # Apply bias if desired
        tmp = inp - self.bias if self.bias is not None else inp
        # Apply weights if desired
        tmp = to.exp(self.log_weight)*tmp if self.log_weight is not None else tmp

        # y = f_nlin( w * (x-b) )
        return self._nonlin(tmp)


class MirrConv1d(_ConvNd):
    """
    Overriding `Conv1d` module implementation from PyTorch 1.4 to re-use parts of the convolution weights by mirroring
    the first half of the kernel (along the columns). This way we can save (close to) half of the parameters, under
    the assumption that we have a kernel that obeys this kind of symmetry.
    The biases are left unchanged.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros'):
        # Same as in PyTorch 1.4
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups,
                         bias, padding_mode)

        # Memorize PyTorch's weight shape (out_channels x in_channels x kernel_size) for later reconstruction
        self.orig_weight_shape = self.weight.shape

        # Get number of kernel elements we later want to use for mirroring
        self.half_kernel_size = ceil(self.weight.shape[2]/2)  # kernel_size = 4 --> 2, kernel_size = 5 --> 3

        # Initialize the weights values the same way PyTorch does
        new_weight_init = to.zeros(self.orig_weight_shape[0], self.orig_weight_shape[1], self.half_kernel_size)
        nn.init.kaiming_uniform_(new_weight_init, a=sqrt(5))

        # Overwrite the weight attribute (transposed is False by default for the Conv1d module and we don't use it here)
        self.weight = nn.Parameter(new_weight_init, requires_grad=True)

    def forward(self, inp: to.Tensor) -> to.Tensor:
        # Reconstruct symmetric weights for convolution (original size)
        mirr_weight = to.empty(self.orig_weight_shape)
        # mirr_weight.fill_(pyrado.inf)  # only for testing

        # Loop over input channels
        for i in range(self.orig_weight_shape[1]):
            # Fill first half
            mirr_weight[:, i, :self.half_kernel_size] = self.weight[:, i, :]

            # Fill second half (flip columns left-right)
            if self.orig_weight_shape[2]%2 == 1:
                # Odd kernel size for convolution, don't flip the last column
                mirr_weight[:, i, self.half_kernel_size:] = to.flip(self.weight[:, i, :], (1,))[:, 1:]
            else:
                # Even kernel size for convolution, flip all columns
                mirr_weight[:, i, self.half_kernel_size:] = to.flip(self.weight[:, i, :], (1,))

        # Only for testing
        # if to.any(to.isinf(mirr_weight)):
        #     raise RuntimeError

        # Run though the same function as the original PyTorch implementation, but with mirrored kernel
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1)//2, self.padding[0]//2)
            return F.conv1d(F.pad(inp, expanded_padding, mode='circular'), mirr_weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(inp, mirr_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
