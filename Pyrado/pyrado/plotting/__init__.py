from pyrado import use_pgf
if use_pgf:
    import matplotlib
    matplotlib.use('pgf')
from matplotlib import font_manager
from matplotlib import pyplot as plt


def set_style(style_name: str = 'default'):
    """
    Sets colors, fonts, font sizes, bounding boxes, and more for plots using pyplot.

    .. note::
        The font sizes of the predefined styles will be overwritten!

    .. seealso::
        https://matplotlib.org/users/customizing.html
        https://matplotlib.org/tutorials/introductory/customizing.html#matplotlib-rcparams
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.rc.html
        https://matplotlib.org/users/usetex.html
        https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex

    :param style_name: str containing the matplotlib style name, or 'default' for the Pyrado default style
    """

    try:
        font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        font_manager.findfont('serif', rebuild_if_missing=False)
    except Exception:
        pass

    if style_name == 'default':
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        plt.rc('text.latex', preamble=r"\usepackage{lmodern}")  # direct font input
        plt.rc('mathtext', fontset='cm')
        plt.rc('pgf', rcfonts=False)  # to use the LaTeX document's fonts in the PGF plots
        plt.rc('image', cmap='inferno')  # default: viridis
        plt.rc('legend', frameon=False)
        plt.rc('legend', framealpha=0.4)
        plt.rc('axes', xmargin=0.)  # disable margins by default
    elif style_name == 'ggplot':
        plt.style.use('ggplot')
    elif style_name == 'dark_background':
        plt.style.use('dark_background')
    elif style_name == 'seaborn':
        plt.style.use('seaborn')
    elif style_name == 'seaborn-muted':
        plt.style.use('seaborn-muted')
    else:
        ValueError("Unknown style name! Got {}, but expected 'default', 'ggplot', 'dark_background',"
                   "'seaborn', or 'seaborn-muted'.".format(style_name))

    plt.rc('font', size=11)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    # plt.rc('savefig', bbox='tight')  # 'tight' is incompatible with pipe-based animation backends
    # plt.rc('savefig', pad_inches=0)


set_style()
