"""
Play around with PyTorch's 1-dim concolution class (in the context of using it for the NFPolicy class)
"""
import torch as to
from matplotlib import pyplot as plt

import pyrado


if __name__ == '__main__':
    pyrado.set_seed(0)

    batch_size = 1
    num_steps = 360
    dim_act = 2
    num_neurons = 4

    kernel_size = 1

    # Create arbitrary signal
    signal = to.zeros(batch_size, dim_act, num_steps)
    signal[:, 0, :] = to.cat([to.zeros(num_steps//2), to.ones(num_steps//2)])
    signal[:, 1, :] = to.cat([to.zeros(num_steps//3), to.ones(num_steps//3), to.zeros(num_steps//3)])
    steps = to.linspace(0, num_steps, steps=num_steps)

    layer = to.nn.Conv1d(in_channels=dim_act, out_channels=num_neurons, kernel_size=kernel_size, stride=1, padding=0,
                         dilation=1, groups=1, bias=False, padding_mode='zeros')
    print(f'layer weights shape: {layer.weight.shape}')
    print(f'layer weights: {layer.weight.data}')

    print(f'input shape:  {signal.shape}')
    with to.no_grad():
        result = layer(signal)
        sum_over_channels = to.sum(result, dim=1, keepdim=True)
    print(f'result shape: {result.shape}')
    print(f'sum_over_channels shape: {sum_over_channels.shape}')

    # Plot
    _, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
    for j in range(dim_act):
        axs[0].plot(steps.numpy(), signal[:, j, :].squeeze(0).numpy(), label=f'signal {j}')

    for k in range(num_neurons):
        axs[1].plot(steps.numpy(), result[:, k, :].squeeze(0).numpy(), label=f'channel {k}')

    axs[2].plot(steps.numpy(), sum_over_channels[:, 0, :].squeeze(0).numpy())

    axs[0].set_ylabel('signal')
    axs[1].set_ylabel('individual convolutions')
    axs[2].set_ylabel('summed over channels')
    axs[0].legend()
    axs[1].legend()
    plt.show()
