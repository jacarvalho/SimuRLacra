import math
import torch as to
from collections import Callable
from torch.optim.optimizer import Optimizer

import pyrado


class GSS(Optimizer):
    """ Golden Section Search optimizer """

    def __init__(self, params, param_min: to.Tensor, param_max: to.Tensor):
        # assert all(group['params'].size() == 1 for group in params)  # only for scalar params
        if not isinstance(param_min, to.Tensor):
            raise pyrado.TypeErr(given=param_min, expected_type=to.Tensor)
        if not isinstance(param_max, to.Tensor):
            raise pyrado.TypeErr(given=param_max, expected_type=to.Tensor)
        if not param_min.shape == param_max.shape:
            raise pyrado.ShapeErr(given=param_min, expected_match=param_max)
        if not all(param_min < param_max):
            raise pyrado.ValueErr(given=param_min, l_constraint=param_max)

        defaults = dict(param_min=param_min, param_max=param_max)
        super().__init__(params, defaults)
        self.gr = (math.sqrt(5) + 1) / 2

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['lb'] = group['param_min']
                state['ub'] = group['param_max']

    def step(self, closure: Callable):
        """
        Performs a single optimization step.

        :param closure: a closure that reevaluates the model and returns the loss
        :return: accumulated loss for all parameter groups after the parameter update
        """
        loss = to.tensor([0.])
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['lb'] = group['param_min']
                    state['ub'] = group['param_max']

                state['step'] += 1

                # Compute new bounds for evaluating
                cand_lb = state['ub'] - (state['ub'] - state['lb']) / self.gr
                cand_ub = state['lb'] + (state['ub'] - state['lb']) / self.gr

                # Set param to lower bound and evaluate
                p.data = cand_lb
                loss_lb = closure()

                # Set param to upper bound and evaluate
                p.data = cand_ub
                loss_ub = closure()

                if loss_lb < loss_ub:
                    state['ub'] = cand_ub
                else:
                    state['lb'] = cand_lb

                # Set param to average value
                p.data = (state['ub'] + state['lb']) / 2.

                # Accumulate the loss
                loss += closure()
        return loss
