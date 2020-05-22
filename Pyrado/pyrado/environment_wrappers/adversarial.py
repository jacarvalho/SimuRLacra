from abc import ABC
import numpy as np
import torch as to

from init_args_serializer import Serializable
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.state_augmentation import StateAugmentationWrapper
from pyrado.environment_wrappers.utils import inner_env, typed_env


class AdversarialWrapper(EnvWrapper, ABC):
    """ Base class for adversarial wrappers (used in ARPL) """

    def __init__(self, wrapped_env, policy, eps, phi):
        EnvWrapper.__init__(self, wrapped_env)
        self._policy = policy
        self._eps = eps
        self._phi = phi

    @staticmethod
    def quadratic_loss(action):
        return to.norm(action).pow(2)

    def decide_apply(self):
        return np.random.binomial(1, self._phi) == 1

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, val):
        self._phi = val


class AdversarialObservationWrapper(AdversarialWrapper, Serializable):
    """" Wrapper to apply adversarial perturbations to the observations (used in ARPL) """

    def __init__(self,
                 wrapped_env,
                 policy,
                 eps,
                 phi):
        """
        Constructor

        :param wrapped_env: environment to be wrapped
        :param policy: policy to be updated
        :param eps: magnitude of perturbation
        :param phi: probability of perturbation
        """
        Serializable._init(self, locals())
        AdversarialWrapper.__init__(self, wrapped_env, policy, eps, phi)

    def step(self, act: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(act)
        adversarial = self.get_arpl_grad(obs)
        if self.decide_apply():
            obs += adversarial.view(-1).float().numpy()
        return obs, reward, done, info

    def get_arpl_grad(self, state):
        state_tensor = to.tensor([state], requires_grad=True, dtype=to.double)
        mean_arpl = self._policy.forward(state_tensor)
        l2_norm_mean = -to.norm(mean_arpl, p=2, dim=1)
        l2_norm_mean.backward()
        state_grad = state_tensor.grad
        return self._eps * to.sign(state_grad)


class AdversarialStateWrapper(AdversarialWrapper, Serializable):
    """" Wrapper to apply adversarial perturbations to the state (used in ARPL) """

    def __init__(self,
                 wrapped_env,
                 policy,
                 eps,
                 phi,
                 torch_observation=False):
        """
        Constructor

        :param wrapped_env: environment to be wrapped
        :param policy: policy to be updated
        :param eps: magnitude of perturbation
        :param phi: probability of perturbation
        :param torch_observation: observation uses torch
        """
        Serializable._init(self, locals())
        AdversarialWrapper.__init__(self, wrapped_env, policy, eps, phi)
        self.torch_observation = torch_observation

    def step(self, act: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(act)
        saw = typed_env(self.wrapped_env, StateAugmentationWrapper)
        state = inner_env(self).state
        nonobserved = to.from_numpy(obs[saw.offset:])
        adversarial = self.get_arpl_grad(state, nonobserved)
        if self.decide_apply():
            inner_env(self).state += adversarial.view(-1).numpy()
        if saw:
            obs[:saw.offset] = inner_env(self).observe(inner_env(self).state)
        else:
            obs = inner_env(self).observe(inner_env(self).state)
        return obs, reward, done, info

    def get_arpl_grad(self, state, nonobserved):
        if isinstance(state, np.ndarray):
            state_tensor = to.tensor(state, requires_grad=True)
        elif isinstance(state, to.Tensor):
            state_tensor = state
        else:
            raise ValueError('state could not be converted to a torch tensor')
        if self.torch_observation:
            observation = inner_env(self).observe(state_tensor, dtype=to.Tensor)
        else:
            observation = state_tensor
        mean_arpl = self._policy.forward(to.cat((observation, nonobserved)))
        l2_norm_mean = -to.norm(mean_arpl, p=2, dim=0)
        l2_norm_mean.backward()
        state_grad = state_tensor.grad
        return self._eps * to.sign(state_grad)


class AdversarialDynamicsWrapper(AdversarialWrapper, Serializable):
    """" Wrapper to apply adversarial perturbations to the domain parameters (used in ARPL) """

    def __init__(self,
                 wrapped_env,
                 policy,
                 eps,
                 phi,
                 width=0.25):
        """
        Constructor

        :param wrapped_env: environemnt to be wrapped
        :param policy: policy to be updated
        :param eps: magnitude of perturbation
        :param phi: probability of perturbation
        :param width: width of distribution to sample from
        """
        Serializable._init(self, locals())
        AdversarialWrapper.__init__(self, wrapped_env, policy, eps, phi)
        self.width = width
        self.saw = typed_env(self.wrapped_env, StateAugmentationWrapper)
        self.nominal = self.saw.nominal
        self.nominalT = to.from_numpy(self.nominal)
        self.adv = None
        self.re_adv()

    def re_adv(self):
        self.adv = np.random.uniform(1 - self.width, 1 + self.width, self.nominal.shape) * self.nominal

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None):
        self.re_adv()
        self.saw.set_param(to.tensor(self.adv))
        return self.wrapped_env.reset(init_state, domain_param)

    def step(self, act: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(act)
        state = obs.clone()
        adversarial = self.get_arpl_grad(state) * self.nominalT
        if self.decide_apply():
            new_params = to.tensor(self.adv).squeeze(0) + adversarial
            self.saw.set_param(new_params.squeeze(0))
        return obs, reward, done, info

    def get_arpl_grad(self, state):
        state_tensor = to.tensor([state], requires_grad=True)
        self.saw.set_param(self.adv)
        mean_arpl = self._policy.forward(state_tensor)
        l2_norm_mean = -to.norm(mean_arpl, p=2, dim=1)
        l2_norm_mean.backward()
        state_grad = state_tensor.grad
        state_grad = state_grad[:, self.saw.offset:]
        return self._eps * to.sign(state_grad)
