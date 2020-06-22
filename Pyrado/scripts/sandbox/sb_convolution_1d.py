"""
Play around with PyTorch's 1-dim concolution class (in the context of using it for the NFPolicy class)
"""
import torch as to
from matplotlib import pyplot as plt

import pyrado
from pyrado.policies.neural_fields import MirrConv1d


if __name__ == '__main__':
    pyrado.set_seed(10)

    hand_coded_filter = False  # use a ramp from 0 to 1 instead of random weights
    use_depth_wise_conv = False
    use_custom_symm_init = True
    # https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    # https://github.com/jayleicn/TVQAplus/blob/master/model/cnn.py
    batch_size = 1
    num_neurons = 360  # each potential-based neuron is basically like time steps of a signal
    in_channels = 1  # number of input signals
    out_channels = 6  # number of filters
    if hand_coded_filter:
        out_channels = 1
    kernel_size = 6  # larger number smooth out and reduce the length of the output signal, use odd numbers
    padding_mode = 'circular'  # circular, reflective, zeros
    padding = kernel_size//2 if padding_mode != 'circular' else kernel_size - 1

    # Create arbitrary signal
    signal = to.zeros(batch_size, in_channels, num_neurons)
    signal[:, 0, :] = to.cat([to.zeros(num_neurons//3), to.ones(num_neurons//3), to.zeros(num_neurons//3)])
    if in_channels == 2:
        signal[:, 1, :] = to.rand_like(signal[:, 0, :])/2

    if use_depth_wise_conv:
        conv_layer = to.nn.Conv1d(in_channels, in_channels, kernel_size, stride=1, padding=padding,
                                  dilation=1, groups=1, bias=False, padding_mode=padding_mode)
        ptwise_conv_layer = to.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')
        print(f'layer weights shape: {conv_layer.weight.shape}')
        print(f'layer2 weights shape: {ptwise_conv_layer.weight.shape}')

    else:
        # Standard way
        conv_layer = to.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                  dilation=1, groups=1, bias=False, padding_mode=padding_mode)
        print(f'layer weights shape: {conv_layer.weight.shape}')

        if hand_coded_filter:
            conv_layer.weight.data = to.linspace(0, 1, kernel_size).repeat(2, 1).unsqueeze(0)

        elif use_custom_symm_init:
            conv_layer = MirrConv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                    dilation=1, groups=1, bias=False, padding_mode=padding_mode)
        print(f'symm layer weights shape: {conv_layer.weight.shape}')

    print(f'input shape:  {signal.shape}')

    with to.no_grad():
        if use_depth_wise_conv:
            result = ptwise_conv_layer(conv_layer(signal))
        else:
            result = conv_layer(signal)

    sum_over_channels = to.sum(result, dim=1, keepdim=True)

    print(f'result shape: {result.shape}')
    print(f'sum_over_channels shape: {sum_over_channels.shape}')

    # Plot
    _, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
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
    plt.show()
