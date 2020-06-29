"""
Collection of plotting functions which take ndarrays as inputs and produce surface plots
"""
import matplotlib as mpl
import numpy as np
import torch as to
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable

import pyrado


def render_surface(
        x: np.ndarray,
        y: np.ndarray,
        z_fcn: [Callable[[np.ndarray], np.ndarray], nn.Module],
        x_label: str,
        y_label: str,
        z_label: str,
        data_format='numpy',
        fig: plt.Figure = None,
        cmap: mpl.cm.ScalarMappable = None,
        title: str = None,
) -> plt.Figure:
    """
    Render a 3-dim surface plot by providing a 1-dim array of x and y points and a function to calculate the z values.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param x: x-axis 1-dim grid for constructing the 2-dim mesh grid
    :param y: y-axis 1-dim grid for constructing the 2-dim mesh grid
    :param z_fcn: function that defines the surface, takes a 2-dim vector as input
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param z_label: label for the z-axis
    :param data_format: data format, 'numpy' or 'torch'
    :param fig: handle to figure, pass None to create a new figure
    :param cmap: color map passed to plot_surface
    :param title: title displayed above the figure, set to None to suppress the title
    :return: handle to figure
    """
    if cmap is None:
        cmap = mpl.rcParams['image.cmap']

    if fig is None:
        fig = plt.figure()
    ax = Axes3D(fig)

    # Create mesh grid matrices from x and y vectors
    xx, yy = np.meshgrid(x, y)

    # Check which version to use based on the output of the function
    if data_format == 'numpy':
        # Operate on ndarrays
        zz = np.array([z_fcn(np.stack((x, y), axis=0)) for x, y in zip(xx, yy)])

    elif data_format == 'torch':
        # Operate on Tensors
        xx_tensor = to.from_numpy(xx)
        yy_tensor = to.from_numpy(yy)

        if hasattr(z_fcn, '_fcn'):
            # Passed function was wrapped (e.g. by functools)
            check_fcn = z_fcn._fcn
        else:
            check_fcn = z_fcn

        if isinstance(check_fcn, nn.Module):
            # Adapt for batch-first behavior of NN-based policies
            zz = to.stack([z_fcn(to.stack((x, y), dim=1).view(-1, 1, 2).to(to.get_default_dtype()))
                           for x, y in zip(xx_tensor, yy_tensor)])
        else:
            zz = to.stack([z_fcn(to.stack((x, y), dim=1).transpose(0, 1).to(to.get_default_dtype()))
                           for x, y in zip(xx_tensor, yy_tensor)])
        zz = zz.squeeze().detach().numpy()

    else:
        raise pyrado.ValueErr(given=data_format, eq_constraint="'numpy' or 'torch'")

    # Generate the plot
    ax.plot_surface(xx, yy, zz, cmap=cmap)

    # Add labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    if title is not None:
        ax.set_title(title)
    return fig
