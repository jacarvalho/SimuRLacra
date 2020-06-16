"""
Play around with PyTorch's 1-dim concolution class (in the context of using it for the NFPolicy class)
"""
import torch as to
from matplotlib import pyplot as plt

import pyrado


if __name__ == '__main__':
    pyrado.set_seed(10)

    use_depth_wise_conv = False
    # https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    # https://github.com/jayleicn/TVQAplus/blob/master/model/cnn.py
    batch_size = 1
    num_neurons = 360  # each potential-based neuron is basically like time steps of a signal
    in_channels = 2  # number of input signals
    out_channels = 4  # number of filters
    kernel_size = 21  # larger number smooth out and reduce the length of the output signal, use odd numbers
    padding = kernel_size//2

    # Create arbitrary signal
    signal = to.zeros(batch_size, in_channels, num_neurons)
    signal[:, 0, :] = to.rand_like(signal[:, 0, :])/2
    signal[:, 1, :] = to.cat([to.zeros(num_neurons//3), to.ones(num_neurons//3), to.zeros(num_neurons//3)])
    steps_in = to.linspace(0, num_neurons, steps=num_neurons)
    steps_out = to.linspace(0, num_neurons - (kernel_size - 1 + 2*padding),
                            steps=num_neurons - (kernel_size - 1) + 2*padding)

    if use_depth_wise_conv:
        conv_layer = to.nn.Conv1d(in_channels, in_channels, kernel_size, stride=1, padding=padding,
                                  dilation=1, groups=1, bias=False, padding_mode='zeros')
        ptwise_conv_layer = to.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')
        print(f'layer weights shape: {conv_layer.weight.shape}')
        print(f'layer2 weights shape: {ptwise_conv_layer.weight.shape}')

    else:
        conv_layer = to.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                  dilation=1, groups=1, bias=False, padding_mode='zeros')
        print(f'layer weights shape: {conv_layer.weight.shape}')

    print(f'layer weights:\n{conv_layer.weight.data.numpy()}')
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
    for b in range(batch_size):
        for j in range(in_channels):
            axs[0].plot(steps_in.numpy(), signal[b, j, :].squeeze(0).numpy(), label=f'signal {j}')

        for k in range(out_channels):
            axs[1].plot(steps_out.numpy(), result[b, k, :].squeeze(0).numpy(), label=f'channel {k}')

        axs[2].plot(steps_out.numpy(), sum_over_channels[b, 0, :].squeeze(0).numpy())

    axs[0].set_ylabel('signal')
    axs[1].set_ylabel('individual convolutions')
    axs[2].set_ylabel('summed over channels')
    axs[0].legend()
    axs[1].legend()
    plt.show()
