"""
Play around with PyTorch's 1-dim concolution class (in the context of using it for the NFPolicy class)

.. seealso::
    # https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    # https://github.com/jayleicn/TVQAplus/blob/master/model/cnn.py
"""
import torch as to
import torch.nn as nn
from matplotlib import pyplot as plt

import pyrado
from pyrado.utils.nn_layers import MirrConv1d


if __name__ == '__main__':
    pyrado.set_seed(10)

    hand_coded_filter = False  # use a ramp from 0 to 1 instead of random weights
    use_depth_wise_conv = False
    use_custom_symm_init = True

    batch_size = 1
    num_neurons = 360  # each potential-based neuron is basically like time steps of a signal
    in_channels = 2  # number of input signals
    out_channels = 6  # number of filters
    if hand_coded_filter:
        out_channels = 1
    kernel_size = 17  # larger number smooth out and reduce the length of the output signal, use odd numbers
    padding_mode = 'circular'  # circular, reflective, zeros
    padding = kernel_size//2 if padding_mode != 'circular' else kernel_size - 1

    # Create arbitrary signal
    signal = to.zeros(batch_size, in_channels, num_neurons)
    signal[:, 0, :] = to.cat([to.zeros(num_neurons//3), to.ones(num_neurons//3), to.zeros(num_neurons//3)])
    if in_channels == 2:
        signal[:, 1, :] = to.rand_like(signal[:, 0, :])/2

    if use_depth_wise_conv:
        conv_layer = nn.Conv1d(in_channels, in_channels, kernel_size, stride=1, padding=padding,
                                  dilation=1, groups=1, bias=False, padding_mode=padding_mode)
        ptwise_conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')
        print(f'conv_layer weights shape: {conv_layer.weight.shape}')
        print(f'ptwise_conv_layer weights shape: {ptwise_conv_layer.weight.shape}')

    else:
        # Standard way
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                  dilation=1, groups=1, bias=False, padding_mode=padding_mode)
        print(f'conv_layer weights shape: {conv_layer.weight.shape}')

        if hand_coded_filter:
            conv_layer.weight.data = to.linspace(0, 1, kernel_size).repeat(2, 1).unsqueeze(0)

        elif use_custom_symm_init:
            conv_layer = MirrConv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                    dilation=1, groups=1, bias=False, padding_mode=padding_mode)
        print(f'mirr conv_layer weights shape: {conv_layer.weight.shape}')

    print(f'input shape:  {signal.shape}')

    with to.no_grad():
        if use_depth_wise_conv:
            result = ptwise_conv_layer(conv_layer(signal))
        else:
            result = conv_layer(signal)

    sum_over_channels = to.sum(result, dim=1, keepdim=True)

    print(f'result shape: {result.shape}')
    print(f'sum_over_channels shape: {sum_over_channels.shape}')

    # Plot signals
    _, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
    colors_in = plt.get_cmap('inferno')(to.linspace(0, 1, in_channels).numpy())
    colors_out = plt.get_cmap('inferno')(to.linspace(0, 1, out_channels).numpy())

    for b in range(batch_size):
        for j in range(in_channels):
            axs[0].plot(signal[b, j, :].squeeze(0).numpy(),  c=colors_in[j])

        for k in range(out_channels):
            axs[1].plot(result[b, k, :].squeeze(0).numpy(), c=colors_out[k])

        axs[2].plot(sum_over_channels[b, 0, :].squeeze(0).numpy())

    axs[0].set_ylabel('input signal')
    axs[1].set_ylabel('individual convolutions')
    axs[2].set_ylabel('summed over channels')

    # Plot weights
    fig = plt.figure(figsize=(8, 12))
    gs = fig.add_gridspec(nrows=out_channels, ncols=in_channels)
    colors_w = plt.get_cmap('inferno')(to.linspace(0, 1, out_channels*in_channels).numpy())

    for j in range(out_channels):
        for k in range(in_channels):
            ax = fig.add_subplot(gs[j, k])
            ax.plot(conv_layer.weight[j, k, :].detach().numpy(), c=colors_w[j*in_channels + k])
            ax.set_xlabel(f'in channel {k}')
            ax.set_ylabel(f'out channel {j}')

    plt.show()
