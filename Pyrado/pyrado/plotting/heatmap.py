import numpy as np
import pandas as pd
from math import floor, ceil
from matplotlib import colorbar
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import ticker
from pandas.core.indexes.numeric import NumericIndex
from typing import Any

import pyrado


def _setup_index_axis(ax: plt.Axes, index: pd.Index, use_index_labels: bool = False, tick_label_precision: int = 3):
    """
    Prepare the index axis for `plot_heatmap`.
    Sets axis label to index name, and axis tick labels to index values.

    :param ax: figure axis to manipulate
    :param index: (pandas DataFrame) index
    :param use_index_labels:
    :param tick_label_precision: floating point precision of the ticks
    """
    if use_index_labels:
        ax.set_label_text(index.name)  # this will clash with using plt.rcParams['use_tex']=True when index contains '_'

    def _index_tick(x, pos):
        """
        Function to define how the ticks on the matplotlib axis should be styled

        :param x: input
        :param pos: unused but necessary for `matplotlib.ticker.FuncFormatter`
        :return: formatted string
        """
        if x.is_integer() or isinstance(x, int):
            # Use the integer value (fixed weird 'close to integer value bug')
            iloc = int(x)
            if iloc < 0 or iloc >= len(index):
                return ""
            tick_val = index[iloc]
        else:
            # Interpolate
            iloc_low = int(floor(x))
            iloc_high = int(ceil(x))
            if iloc_low < 0 or iloc_high >= len(index):
                return ""
            low = index[iloc_low]
            high = index[iloc_high]

            tick_val = low*(x - iloc_low) + high*(1 - x + iloc_low)

        # Format tick value
        return f'{tick_val:.{tick_label_precision}f}'

    # Apply tick format
    ax.set_major_formatter(ticker.FuncFormatter(_index_tick))  # index[int(x)]))
    ax.set_minor_formatter(ticker.NullFormatter())


def _annotate_heatmap(img,
                      data=None,
                      valfmt: str = '{x:.2f}',
                      textcolors: tuple = ('black', 'white'),
                      thold: float = None,
                      **textkw: Any):
    """
    Annotate a given heat map.
    .. note:: The text color changes based on a threshold which only makes sense for color maps going from dark to bright.

    :param img: AxesImage to be labeled.
    :param data: data used to annotate. If None, the image's data is used.
    :param valfmt: format of the annotations inside the heat map. This should either use the string format method, e.g.
                   '$ {x:.2f}', or be a :class:matplotlib.ticker.Formatter.
    :param textcolors: list or array of two color specifications. The first is used for values below a threshold,
                       the second for those above.
    :param thold: threshold value in data units according to which the colors from textcolors are applied.
                  If None (by default) uses the middle of the colormap as separation.
    :param textkw: further arguments passed on to the created text labels
    """
    if not isinstance(data, (list, np.ndarray)):
        data = img.get_array()

    # Normalize the threshold to the images color range
    thold = img.norm(thold) if thold is not None else img.norm(data.max())/2.

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
            kw.update(color=textcolors[img.norm(data[i, j]) < thold])  # if true then use second color
            text = img.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)


def render_heatmap(
    data: pd.DataFrame,
    ax_hm: plt.Axes = None,
    cmap: colors.Colormap = None,
    norm: colors.Normalize = colors.Normalize(),
    annotate: bool = True,
    annotation_valfmt: str = '{x:.0f}',
    add_sep_colorbar: bool = False,
    ax_cb: plt.Axes = None,
    colorbar_label: str = None,
    use_index_labels: bool = False,
    xlabel: str = None,
    ylabel: str = None,
    fig_canvas_title: str = None,
    fig_size: tuple = (8, 6),
    manipulate_ticks: bool = False,
    tick_label_prec: int = 3,
    xtick_label_prec: int = None,
    ytick_label_prec: int = None
) -> (plt.Figure, plt.Figure):
    """
    Plot a 2D heat map from a 2D `pandas.DataFrame` using pyplot.
    The data frame should have exactly one column index level and one row index level. These will automatically become
    the axis ticks. It is assumed that the data is equally spaced.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param data: 2D pandas DataFrame
    :param ax_hm: axis to draw the heat map onto, if `None` a new figure is created
    :param cmap: colormap passed to imshow
    :param norm: colormap normalizer passed to imshow
    :param annotate: select if the heat map should be annotated
    :param annotation_valfmt: format of the annotations inside the heat map, irrelevant if annotate = False
    :param add_sep_colorbar: flag if a separate color bar is added automatically
    :param ax_cb: axis to draw the color bar onto, if `None` a new figure is created
    :param colorbar_label: label for the color bar
    :param use_index_labels: flag if index names from the pandas DataFrame are used as labels for the x- and y-axis.
                             This can can be overridden by xlabel and ylabel.
    :param xlabel: label for the x axis
    :param ylabel: label for the y axis
    :param fig_canvas_title: window title for the heat map plot, no title by default
    :param fig_size: width and height of the figure in inches
    :param manipulate_ticks: apply custom manipulation to the x and y axis ticks
    :param tick_label_prec: floating point precision of the x- and y-axis labels.
                            This can be overwritten xtick_label_prec and ytick_label_prec
    :param xtick_label_prec: floating point precision of the x-axis labels
    :param ytick_label_prec: floating point precision of the y-axis labels
    :return: handles to the heat map and the color bar figures (None if not existent)
    """
    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, NumericIndex):
            raise pyrado.TypeErr(given=data.index, expected_type=NumericIndex)
        if not isinstance(data.columns, NumericIndex):
            raise pyrado.TypeErr(given=data.columns, expected_type=NumericIndex)
        # Extract the data
        x = data.columns
        y = data.index
    else:
        raise pyrado.TypeErr(given=data, expected_type=pd.DataFrame)

    # Create axes if not provided
    if ax_hm is None:
        fig_hm, ax_hm = plt.subplots(1, figsize=fig_size)
    else:
        fig_hm = ax_hm.figure

    if fig_canvas_title is not None:
        fig_hm.canvas.set_window_title(fig_canvas_title)

    # Create the image
    img = ax_hm.imshow(data, cmap=cmap, norm=norm, aspect=(x.max()-x.min())/(y.max()-y.min()), origin='lower',
                       extent=[x.min(), x.max(), y.min(), y.max()])  # former: aspect='auto'

    # Set axes limits
    ax_hm.set_xlim(x.min(), x.max())
    ax_hm.set_ylim(y.min(), y.max())

    # Annotate the heat map
    if annotate:
        _annotate_heatmap(img, valfmt=annotation_valfmt)

    # Prepare the ticks
    if manipulate_ticks:
        _setup_index_axis(ax_hm.xaxis, x, use_index_labels,
                          xtick_label_prec if xtick_label_prec is not None else tick_label_prec)
        _setup_index_axis(ax_hm.yaxis, y, use_index_labels,
                          ytick_label_prec if ytick_label_prec is not None else tick_label_prec)

    ax_hm.stale = True  # to cause redraw

    # Set the labels
    if xlabel is not None:
        ax_hm.set_xlabel(xlabel)
    if ylabel is not None:
        ax_hm.set_ylabel(ylabel)

    # Add color bar if requested
    if add_sep_colorbar:
        # Draw a new figure and re-plot the color bar there
        if ax_cb is None:
            fig_cb, ax_cb = plt.subplots(1, figsize=fig_size)
        else:
            fig_cb = plt.gcf()

        if colorbar_label is not None:
            colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, label=colorbar_label)
        else:
            colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)
        # if colorbar_label is not None:
        #     fig_cb.colorbar(img, ax=ax_cb, label=colorbar_label)  # plt.colorbar works, too
        # else:
        #     fig_cb.colorbar(img, ax=ax_cb)  # plt.colorbar works, too
        #
        # # Only show the color bar
        # ax_cb.remove()
        return fig_hm, fig_cb

    else:
        return fig_hm, None
