# Import everything from the C extension
from _rcsenv import *

# Expose config folder
import os.path as osp

RCSPYSIM_CONFIG_PATH = osp.join(osp.dirname(__file__), "config")

__all__ = [
    "BoxSpace",
    "RcsSimEnv",
    "ControlPolicy",
    "MLPPolicy",
    "setLogLevel",
    "addResourcePath",
    "setVortexLogLevel",
    "RCSPYSIM_CONFIG_PATH",
    "JointLimitException",
]
