import os.path as osp
from matplotlib import pyplot as plt
from typing import Callable, Any


class _LFMEntry:
    """ One plot managed by the LiveFigureManager """

    def __init__(self, update_function, title):
        self.update_function = update_function
        self.title = title
        self._fignum = None

    def update(self, data, args):
        """
        Update plot.

        :param data: data to plot
        :param args: parsed command line arguments
        """
        if self._fignum is None:
            fig = plt.figure(num=self.title)
            self._fignum = fig.number
        elif not plt.fignum_exists(self._fignum):
            # got closed
            return False
        else:
            fig = plt.figure(self._fignum)

        fig.clf()
        # call drawer
        res = self.update_function(fig, data, args)

        # Signal that we're still alive
        if res is False:
            # Cancelled, close figure
            plt.close(fig)
            return False

        return True


class LiveFigureManager:
    """
    Manages multiple matplotlib figures and refreshes them when the input file changes.
    It also ensures that if you close a figure, it does not reappear on the next update.
    If all figures are closed, the update loop is stopped.
    """

    def __init__(self, file_path: str, data_loader: Callable[[str], Any], args, update_interval: int = 3):
        """
        Constructor

        :param file_path: name of file to load updates from
        :param data_loader: called to load the file contents into some internal representation like a pandas `DataFrame`
        :param args: parsed command line arguments
        :param update_interval: time to wait between figure updates [s]
        """
        self._file_path = file_path
        self._data_loader = data_loader
        self._args = args
        self._update_interval = update_interval
        self._figure_list = []

    def figure(self, title: str = None):
        """
        Decorator to define a figure update function.
        Every marked function will be called when the file changes to visualize the updated data.

        :usage:
        .. code-block:: python

            @lfm.figure('A figure')
            def a_figure(fig, data, args):
                ax = fig.add_subplot(111)
                ax.plot(data[...])

        :param title: figure title
        :return: decorator for the plotting function
        """

        def wrapper(func):
            entry = _LFMEntry(func, title)
            self._figure_list.append(entry)
            return entry

        return wrapper

    def _plot_all(self):
        data = self._data_loader(self._file_path)
        self._figure_list[:] = [pl for pl in self._figure_list if pl.update(data, self._args)]

    def spin(self):
        """ Run the plot update loop.  """
        plt.ion()
        # Create all plot
        self._plot_all()

        # Watch modification time
        time_last_plot = osp.getmtime(self._file_path)

        while len(plt.get_fignums()) > 0:
            # Check for changes
            mt = osp.getmtime(self._file_path)

            if mt > time_last_plot:
                # Changed, so update
                self._plot_all()
                time_last_plot = mt

            # Give matplotlib some time
            plt.pause(self._update_interval)
