import torch as to
from torch.distributions.uniform import Uniform

from pyrado.policies.base import Policy
from pyrado.policies.base_recurrent import RecurrentPolicy
from pyrado.utils.data_types import EnvSpec


class IdlePolicy(Policy):
    """ The most simple policy which simply does nothing """

    name: str = 'idle'

    def __init__(self, spec: EnvSpec, use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor = None) -> to.Tensor:
        # Observations are ignored
        return to.zeros(self._env_spec.act_space.shape)


class DummyPolicy(Policy):
    """ Simple policy which samples random values form the action space """

    name: str = 'dummy'

    def __init__(self, spec: EnvSpec, use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        low = to.from_numpy(spec.act_space.bound_lo)
        high = to.from_numpy(spec.act_space.bound_up)
        self._distr = Uniform(low, high)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor = None) -> to.Tensor:
        # Observations are ignored
        return self._distr.sample()


class RecurrentDummyPolicy(RecurrentPolicy):
    """
    Simple recurrent policy which samples random values form the action space and
    always returns hidden states with value zero
    """

    name: str = 'rec_cummy'

    def __init__(self, spec: EnvSpec, hidden_size: int, use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param hidden_size: size of the mimic hidden layer
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        low = to.from_numpy(spec.act_space.bound_lo)
        high = to.from_numpy(spec.act_space.bound_up)
        self._distr = Uniform(low, high)
        self._hidden_size = hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor = None, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        # Observations and hidden states are ignored
        return self._distr.sample(), to.zeros(self._hidden_size)
