import numpy as np
from matplotlib import colors as colors


class AccNorm(colors.Normalize):
    """
    Accumulative normalizer which is useful to have one colormap consistent for multiple images.
    Adding new data will expand the limits.
    """

    def autoscale(self, A):
        # Also update values if scale expands
        vmin = np.min(A)
        if self.vmin is None or self.vmin > vmin:
            self.vmin = vmin

        vmax = np.max(A)
        if self.vmax is None or self.vmax < vmax:
            self.vmax = vmax

    def autoscale_None(self, A):
        # Also update values if scale expands
        vmin = np.min(A)
        if self.vmin is None or self.vmin > vmin:
            self.vmin = vmin

        vmax = np.max(A)
        if self.vmax is None or self.vmax < vmax:
            self.vmax = vmax
