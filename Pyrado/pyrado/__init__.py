import numpy as np
import os.path as osp
import random
import torch as to
from colorama import init
from os import _exit

# Provide global data directories
PERMA_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'perma')
EVAL_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'perma', 'evaluation')
EXP_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'perma', 'experiments')
HPARAM_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'perma', 'hyperparams')
TEMP_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'temp')
MUJOCO_ASSETS_DIR = osp.join(osp.dirname(__file__), 'environments', 'mujoco', 'assets')
ISAAC_ASSETS_DIR = osp.join(osp.dirname(__file__), '..', '..', 'thirdParty', 'isaac_gym', 'assets')

# Check if the interfaces to the physics engines are available
try:
    import rcsenv
except ImportError:
    rcsenv_available = False
else:
    rcsenv_available = True

try:
    import mujoco_py
except (ImportError, Exception):
    # The ImportError is raised if mujoco-py is simply not installed
    # The Exception catches the case that you have everything installed properly but your IDE does not set the
    # LD_LIBRARY_PATH correctly (happens for PyCharm & CLion). To check this, try to run your script from the terminal.
    mujoco_available = False
else:
    mujoco_available = True

# Set default data type for PyTorch
to.set_default_dtype(to.double)

# Convenient math variables
inf = float('inf')
nan = float('nan')

# Figure sizes (width, height) [inch]
figsize_thesis_2percol_18to10 = (2.9, 2.9/18*10)
figsize_thesis_2percol_16to10 = (2.9, 2.9/16*10)
figsize_IEEE_1col_18to10 = (3.5, 3.5/18*10)
figsize_IEEE_2col_18to10 = (7.16, 7.16/18*10)
figsize_IEEE_1col_square = (3.5, 3.5)
figsize_IEEE_2col_square = (7.16, 7.16)
figsize_JMLR_warpfig = (2.5, 2.4)

# Set style for printing and plotting
use_pgf = False
from pyrado import plotting

# Reset the colorama style after each print
init(autoreset=True)

# Set a uniform printing style for PyTorch
to.set_printoptions(precision=4, linewidth=200)

# Set a uniform printing style for numpy
np.set_printoptions(precision=4, sign=' ', linewidth=200)  # suppress=True

# Include all error classes
from pyrado.utils.exceptions import BaseErr, ValueErr, PathErr, ShapeErr, TypeErr

# Set the public API
__all__ = ['TEMP_DIR', 'PERMA_DIR', 'EVAL_DIR', 'EXP_DIR', 'HPARAM_DIR',
           'rcsenv_available', 'mujoco_available', 'use_pgf', 'inf', 'nan']


def close_vpython():
    """ Forcefully close the connection to the current VPython animation """
    _exit(0)


def set_seed(seed: int):
    """
    Set the seed for the random number generators

    :param seed: value for the random number generators' seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    to.manual_seed(seed)
    if to.cuda.is_available():
        to.cuda.manual_seed_all(seed)
