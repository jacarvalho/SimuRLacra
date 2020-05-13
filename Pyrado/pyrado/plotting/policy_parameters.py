"""
Functions to plot Pyrado policies
"""
import numpy as np
import torch as to
from matplotlib import ticker, gridspec, colorbar
from matplotlib import pyplot as plt
from typing import Any

import pyrado
from pyrado.plotting.utils import AccNorm
from pyrado.policies.base import Policy
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import ensure_no_subscript, ensure_math_mode


def _annotate_img(img,
                  data: [list, np.ndarray] = None,
                  thold_lo: float = None,
                  thold_up: float = None,
                  valfmt: str = '{x:.2f}',
                  textcolors: tuple = ('white', 'black'),
                  **textkw: Any):
    """
    Annotate a given image.

    .. note::
        The text color changes based on thresholds which only make sense for symmetric color maps.

    :param mg: AxesImage to be labeled.
    :param data: data used to annotate. If None, the image's data is used.
    :param thold_lo: lower threshold for changing the color
    :param thold_up: upper threshold for changing the color
    :param valfmt: format of the annotations inside the heat map. This should either use the string format method, e.g.
                   '$ {x:.2f}', or be a :class:matplotlib.ticker.Formatter.
    :param textcolors: two color specifications. The first is used for values below a threshold,
                       the second for those above.
    :param textkw: further arguments passed on to the created text labels
    """
    if not isinstance(data, (list, np.ndarray)):
        data = img.get_array()

    # Normalize the threshold to the images color range
    if thold_lo is None:
        thold_lo = data.min()*0.5
    if thold_up is None:
        thold_up = data.max()*0.5

    # Set default alignment to center, but allow it to be overwritten by textkw
    kw = dict(horizontalalignment='center', verticalalignment='center')
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a text for each 'pixel'.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[thold_lo < data[i, j] < thold_up])  # if true then use second color
            text = img.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)


def render_policy_params(policy: Policy,
                         env_spec: EnvSpec,
                         cmap_name: str = 'RdBu',
                         ax_hm: plt.Axes = None,
                         annotate: bool = True,
                         annotation_valfmt: str = '{x:.2f}',
                         colorbar_label: str = '',
                         xlabel: str = None,
                         ylabel: str = None,
                         ) -> plt.Figure:
    """
    Plot the weights and biases as images, and a color bar.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param policy: policy to visualize
    :param env_spec: environment specification
    :param cmap_name: name of the color map, e.g. 'inferno', 'RdBu', or 'viridis'
    :param ax_hm: axis to draw the heat map onto, if equal to None a new figure is opened
    :param annotate: select if the heat map should be annotated
    :param annotation_valfmt: format of the annotations inside the heat map, irrelevant if annotate = False
    :param colorbar_label: label for the color bar
    :param xlabel: label for the x axis
    :param ylabel: label for the y axis
    :return: handles to figures
    """
    if not isinstance(policy, to.nn.Module):
        raise pyrado.TypeErr(given=policy, expected_type=to.nn.Module)
    cmap = plt.get_cmap(cmap_name)

    # Create axes and subplots depending on the NN structure
    num_rows = len(list(policy.parameters()))
    fig = plt.figure(figsize=(14, 10), tight_layout=False)
    gs = gridspec.GridSpec(num_rows, 2, width_ratios=[14, 1])  # right column is the color bar
    ax_cb = plt.subplot(gs[:, 1])

    # Accumulative norm for the colors
    norm = AccNorm()

    for i, (name, param) in enumerate(policy.named_parameters()):
        # Create current axis
        ax = plt.subplot(gs[i, 0])
        ax.set_title(name.replace('_', '\_'))

        # Convert the data and plot the image with the colors proportional to the parameters
        data = np.atleast_2d(param.detach().numpy())
        img = plt.imshow(data, cmap=cmap, norm=norm, aspect='auto', origin='lower')

        if annotate:
            _annotate_img(img, thold_lo=min(policy.param_values)*0.75, thold_up=max(policy.param_values)*0.75,
                          valfmt=annotation_valfmt)

        # Prepare the ticks
        if name == 'obs_layer.weight':
            # Set the labels in case of an ADN policy
            ax.set_xticks(np.arange(env_spec.obs_space.flat_dim))
            ax.set_yticks(np.arange(env_spec.act_space.flat_dim))
            ax.set_xticklabels(ensure_no_subscript(env_spec.obs_space.labels))
            ax.set_yticklabels(ensure_math_mode(env_spec.act_space.labels))
        elif name in ['obs_layer.bias', 'scaling_layer.log_weight']:
            # Set the labels in case of an ADN policy
            ax.set_xticks(np.arange(env_spec.act_space.flat_dim))
            ax.set_xticklabels(ensure_math_mode(env_spec.act_space.labels))
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        elif name == 'prev_act_layer.weight':
            # Set the labels in case of an ADN policy
            ax.set_xticks(np.arange(env_spec.act_space.flat_dim))
            ax.set_yticks(np.arange(env_spec.act_space.flat_dim))
            ax.set_xticklabels(ensure_math_mode(env_spec.act_space.labels))
            ax.set_yticklabels(ensure_math_mode(env_spec.act_space.labels))
        elif name in ['_log_tau', '_log_kappa']:
            # Set the labels in case of an ADN policy
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            # ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            # ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        # Add the color bar (call this within the loop to make the AccNorm scan every image)
        colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, label=colorbar_label)

    # Increase the vertical white spaces between the subplots
    plt.subplots_adjust(hspace=.7, wspace=0.1)

    # Set the labels
    if xlabel is not None:
        ax_hm.set_xlabel(xlabel)
    if ylabel is not None:
        ax_hm.set_ylabel(ylabel)

    return fig
