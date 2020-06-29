import inspect
import torch as to
from typing import Callable, List
from torch.jit import ScriptModule, script, export
from torch.nn import Module

from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy


class TimePolicy(Policy):
    """ A purely time-based policy, mainly useful for testing """

    name: str = 'time'

    def __init__(self, spec: EnvSpec, fcn_of_time: Callable[[float], List[float]], dt: float, use_cuda: bool = False):
        """
        Constructor

        :usage:
        .. code-block:: python

            policy = TimePolicy(env, lambda t: to.tensor([-sin(t) * 0.001]), 0.01)

        :param spec: environment specification
        :param fcn_of_time: time-depended function returning actions
        :param dt: time step [s]
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        # Script the function eagerly
        self._fcn_of_time = fcn_of_time
        self._dt = dt
        self._curr_time = None

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def reset(self):
        self._curr_time = 0

    def forward(self, obs: to.Tensor) -> to.Tensor:
        act = to.tensor(self._fcn_of_time(self._curr_time), dtype=to.get_default_dtype())
        self._curr_time += self._dt
        return act

    def trace(self) -> ScriptModule:
        return script(TraceableTimePolicy(self.env_spec, self._fcn_of_time, self._dt))


class TraceableTimePolicy(Module):
    """
    A scriptable version of TimePolicy.

    We could try to make TimePolicy itself scriptable, but that won't work anyways due to Policy not being scriptable.
    Better to just write another class.
    """

    # Attributes
    input_size: int
    output_size: int
    dt: float
    current_time: float

    def __init__(self, spec: EnvSpec, fcn_of_time: Callable[[float], List[float]], dt: float):
        super().__init__()

        # Setup attributes
        self.input_size = spec.obs_space.flat_dim
        self.output_size = spec.act_space.flat_dim
        self.dt = dt

        self.env_spec = spec

        # Validate function signature
        sig = inspect.signature(fcn_of_time, follow_wrapped=False)
        posp = [p for p in sig.parameters.values() if p.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }]
        assert len(posp) == 1
        param = next(iter(posp))
        # check parameter type
        if param.annotation != float:
            raise ValueError(f"Malformed fcn_of_time with signature {sig} - parameter must have type float")
        # check return type
        if sig.return_annotation != inspect.Signature.empty and sig.return_annotation != List[float]:
            raise ValueError(f"Malformed fcn_of_time with signature {sig} - return type must be List[float]")

        self.fcn_of_time = fcn_of_time

        # setup current time buffer
        self.current_time = 0.

    @export
    def reset(self):
        self.current_time = 0.

    def forward(self, obs_ignore):
        act = to.tensor(self.fcn_of_time(self.current_time), dtype=to.double)
        self.current_time = self.current_time + self.dt
        return act
