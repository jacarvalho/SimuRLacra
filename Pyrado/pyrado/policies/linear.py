import torch as to

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy
from pyrado.policies.features import FeatureStack
from pyrado.policies.initialization import init_param


class LinearPolicy(Policy):
    """
    A linear policy defined by the inner product of nonlinear features of the observations with the policy parameters
    """

    def __init__(self, spec: EnvSpec, feats: FeatureStack, init_param_kwargs: dict = None, use_cuda: bool = False):
        """
        Constructor

        :param spec: specification of environment
        :param feats: list of feature functions
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        """
        super().__init__(spec, use_cuda)

        if not isinstance(feats, FeatureStack):
            raise pyrado.TypeErr(given=feats, expected_type=FeatureStack)

        # Store inputs
        self._num_act = spec.act_space.flat_dim
        self._num_obs = spec.obs_space.flat_dim

        self._feats = feats
        self.num_active_feat = feats.get_num_feat(self._num_obs)
        self.net = to.nn.Linear(self.num_active_feat, self._num_act, bias=False)

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    @property
    def features(self) -> FeatureStack:
        """ Get the (nonlinear) feature transformations. """
        return self._feats

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Initialize the linear layer using default initialization
            init_param(self.net, **kwargs)
        else:
            self.param_values = init_values  # ignore the IntelliJ warning

    def eval_feats(self, obs: to.Tensor) -> to.Tensor:
        """
        Evaluate the features for the given observations.

        :param obs: observation from the environment
        :return feats_val: the features' values
        """
        return self._feats(obs)

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Evaluate the features at the given observation or use given feature values

        :param obs: observations from the environment
        :return: actions
        """
        obs = obs.to(self.device)
        batched = obs.ndimension() == 2  # number of dim is 1 if unbatched, dim > 2 is cought by features
        feats_val = self.eval_feats(obs)

        # Inner product between policy parameters and the value of the features
        act = self.net(feats_val)

        # Return the flattened tensor if not run in a batch mode to be compatible with the action spaces
        return act.flatten() if not batched else act
