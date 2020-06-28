import torch as to
import torch.nn as nn
import torch.nn.init as init
from math import sqrt, ceil
from warnings import warn

import pyrado
from pyrado.utils.nn_layers import ScaleLayer, PositiveScaleLayer, IndiNonlinLayer


def init_param(m, **kwargs):
    """
    Initialize the parameters of the PyTorch Module / layer / network / cell according to its type.

    :param m: PyTorch Module / layer / network / cell to initialize
    :param kwargs: optional keyword arguments, e.g. `t_max` for LSTM's chrono initialization [2]

    .. seealso::
        [1] A.M. Sachse, J. L. McClelland, S. Ganguli, "Exact solutions to the nonlinear dynamics of learning in
        deep linear neural networks", 2014

        [2] C. Tallec, Y. Ollivier, "Can recurrent neural networks warp time?", 2018, ICLR
    """
    kwargs = kwargs if kwargs is not None else dict()

    if isinstance(m, nn.Linear):
        if m.weight.data.ndimension() >= 2:
            # Most common case
            init.orthogonal_(m.weight.data)  # former: init.xavier_normal_(m.weight.data)
        else:
            warn('Orthogonal initialization is not possible for tensors with less than 2 dimensions.'
                 'Falling back to Gaussian initialization.')
            init.normal_(m.weight.data)

        if m.bias is not None:
            if kwargs.get('uniform_bias', False):
                init.uniform_(m.bias.data, a=-1./sqrt(m.bias.data.nelement()), b=1./sqrt(m.bias.data.nelement()))
            else:
                # Most common case
                init.normal_(m.bias.data)

    elif isinstance(m, nn.RNN):
        for param in m.parameters():
            if len(param.shape) >= 2:
                # Most common case
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                # Most common case
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # Initialize the input to hidden weights orthogonally
                # w_ii, w_if, w_ic, w_io
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                # Initialize the hidden to hidden weights separately as identity matrices and stack them afterwards
                # w_ii, w_if, w_ic, w_io
                weight_hh_ii = to.eye(m.hidden_size, m.hidden_size)
                weight_hh_if = to.eye(m.hidden_size, m.hidden_size)
                weight_hh_ic = to.eye(m.hidden_size, m.hidden_size)
                weight_hh_io = to.eye(m.hidden_size, m.hidden_size)
                weight_hh_all = to.cat([weight_hh_ii, weight_hh_if, weight_hh_ic, weight_hh_io], dim=0)
                param.data.copy_(weight_hh_all)
            elif 'bias' in name:
                # b_ii, b_if, b_ig, b_io
                if 't_max' in kwargs:
                    if not isinstance(kwargs['t_max'], (float, int, to.Tensor)):
                        raise pyrado.TypeErr(given=kwargs['t_max'], expected_type=[float, int, to.Tensor])
                    # Initialize all biases to 0, but the bias of the forget and input gate using the chrono init
                    nn.init.constant_(param.data, val=0)
                    param.data[m.hidden_size:m.hidden_size*2] = to.log(nn.init.uniform_(  # forget gate
                        param.data[m.hidden_size:m.hidden_size*2], 1, kwargs['t_max'] - 1
                    ))
                    param.data[0: m.hidden_size] = -param.data[m.hidden_size: 2*m.hidden_size]  # input gate
                else:
                    # Initialize all biases to 0, but the bias of the forget gate to 1
                    nn.init.constant_(param.data, val=0)
                    param.data[m.hidden_size:m.hidden_size*2].fill_(1)

    elif isinstance(m, nn.Conv1d):
        # Not implemented
        pass

    elif isinstance(m, ScaleLayer):
        # Initialize all weights to 1
        m.weight.data.fill_(1.)

    elif isinstance(m, PositiveScaleLayer):
        # Initialize all weights to 1
        m.log_weight.data.fill_(0.)

    elif isinstance(m, IndiNonlinLayer):
        # Initialize all weights to 1 and all biases (if they exist) to 0
        m.log_weight.data.fill_(0.)
        if m.bias is None:
            pass
        else:
            m.bias.data.fill_(0.)

    else:
        pass
