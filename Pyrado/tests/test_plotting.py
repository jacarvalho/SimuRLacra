import pytest
import numpy as np

from matplotlib import pyplot as plt
from pyrado.utils.functions import rosenbrock
from pyrado.plotting.surface import render_surface


@pytest.mark.visualization
@pytest.mark.parametrize(
        'x, y, data_format', [
                (np.linspace(-2, 2, 30, True), np.linspace(-1, 3, 30, True), 'numpy'),
                (np.linspace(-2, 2, 30, True), np.linspace(-1, 3, 30, True), 'torch'),
        ], ids=['numpy', 'torch']
)
def test_surface(x, y, data_format):
    render_surface(x, y, rosenbrock, 'x', 'y', 'z', data_format)
    plt.show()
