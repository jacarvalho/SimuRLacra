"""
Script to plot a value function as a surface over a gird of values for 2 selected dimensions of the state
"""
import numpy as np
import os.path as osp
import torch as to
from matplotlib import pyplot as plt
from warnings import warn

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.surface import render_surface
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import ensure_math_mode


class wrap_value_fcn:
    """ Wrap the values function to be able to only pass a subset of the state. """

    def __init__(self, fcn: nn.Module, fixed_state: to.Tensor, idcs: list):
        """
        Constructor

        :param fcn: function to wrap with an input dimension >= `len(args.idcs)`
        :param fixed_state: state values held constant for the evaluation, dimension matches the Module's input layer
        :param idcs: indices of the state dimensions where the `fixed_state` is replaced which values from outer scope
        """
        self._fcn = fcn
        self._fixed_state = fixed_state
        self._idcs = idcs

    def __call__(self, varying: to.Tensor) -> to.Tensor:
        """
        Call the value function wrapper

        :param varying: the changing part of the state
        :return: function value
        """
        state = self._fixed_state.clone()
        # First dimension is the batch size, second dimension is 1, third dimension is the full state dimension
        state = state.repeat(varying.shape[0], varying.shape[1], 1)
        # Insert the values of the evaluation mesh grid into the selected state dimensions
        state[:, :, self._idcs] = varying
        return self._fcn(state)


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Load the environment and the value function
    env, _, kwout = load_experiment(ex_dir, args)
    value_fcn = kwout['valuefcn']

    if not len(args.idcs) == 2:
        pyrado.ShapeErr(msg='Please provide exactly two indices to slice the value function input space (obs_space)!')

    # Use the environments lower and upper bounds to parametrize the mesh grid
    lb, ub = env.obs_space.bounds
    lb_inf_check = np.isinf(lb)
    ub_inf_check = np.isinf(ub)
    if lb_inf_check.any():
        warn("Detected at least one inf entry in mesh grid's lower bound, replacing all with -1.")
        lb[lb_inf_check] = -1.
    if ub_inf_check.any():
        warn("Detected at least one inf entry in mesh grid's upper bound, replacing all with 1.")
        ub[ub_inf_check] = 1.

    x = np.linspace(lb[args.idcs[0]], ub[args.idcs[0]], 20)
    y = np.linspace(lb[args.idcs[1]], ub[args.idcs[1]], 20)

    # Create labels for the plot based on the labels of the environment
    space_labels = env.obs_space.labels
    if space_labels is not None:
        state_labels = [space_labels[args.idcs[0]], space_labels[args.idcs[1]]]
    else:
        state_labels = ['s_' + str(args.idcs[0]), 's_' + str(args.idcs[1])]

    # Provide the state at which the value function should be evaluated (partially overwritten by the evaluation gird)
    fixed_state = to.zeros(env.obs_space.shape)

    # Wrap the function to be able to only provide the mesh gird values as inputs
    w_value_fcn = wrap_value_fcn(value_fcn, fixed_state, args.idcs)

    fig = render_surface(to.from_numpy(x), to.from_numpy(y), w_value_fcn,
                         f'{ensure_math_mode(state_labels[0])}', f'{ensure_math_mode(state_labels[1])}',
                         f'$V(${ensure_math_mode(state_labels[0])},{ensure_math_mode(state_labels[1])}$)$',
                         data_format='torch')

    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(ex_dir, f'valuefcn-{state_labels[0]}-{state_labels[1]}.{fmt}'), dpi=500)

    plt.show()
