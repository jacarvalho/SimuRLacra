import numpy as np
import torch as to
import torch.distributions as torchdist
from copy import deepcopy

from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.algorithms.torchdist_utils import DoubleSidedStandardMaxwell, std_gaussian_from_std_dsmaxwell
from pyrado.environments.base import Env
from pyrado.policies.base import Policy
from pyrado.utils.math import clamp_symm
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult
from pyrado.exploration.stochastic_params import SymmParamExplStrat, NormalParamNoise
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode


class EMVD(ParameterExploring):
    """
    Episodic Measure-Valued Derivatives (E-MVD)

    """

    name: str = 'mvd'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 distribution,
                 max_iter: int,
                 num_rollouts: int,
                 expl_std_init: float,
                 expl_std_min: float = 0.01,
                 pop_size: int = None,
                 clip_ratio_std: float = 0.05,
                 normalize_update: bool = False,
                 transform_returns: bool = True,
                 num_sampler_envs: int = 4,
                 n_mc_samples_gradient=1,
                 coupling=True,
                 real_env=False,
                 lr: float = 5e-4,
                 optim: str = 'SGD',
                 base_seed: int = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param pop_size: number of solutions in the population
        :param num_rollouts: number of rollouts per policy sample
        :param expl_std_init: initial standard deviation for the exploration strategy
        :param expl_std_min: minimal standard deviation for the exploration strategy
        :param clip_ratio_std: maximal ratio for the change of the exploration strategy's standard deviation
        :param transform_returns: use a rank-transformation of the returns to update the policy
        :param base_seed: seed added to all other seeds in order to make the experiments distinct but repeatable
        """
        # Call ParameterExploring's constructor
        super().__init__(
            save_dir,
            env,
            policy,
            max_iter,
            num_rollouts,
            pop_size=pop_size,
            base_seed=base_seed,
            num_sampler_envs=num_sampler_envs,
        )

        self._distribution = distribution
        self._dims = distribution.get_number_of_dims()

        self._n_mc_samples_gradient = n_mc_samples_gradient
        self._coupling = coupling

        self._real_env = real_env

        # Store the inputs
        self.clip_ratio_std = clip_ratio_std
        self.normalize_update = normalize_update
        self.transform_returns = transform_returns
        self.lr = lr

        # Exploration strategy based on symmetrical normally distributed noise
        if self.pop_size%2 != 0:
            # Symmetric buffer needs to have an even number of samples
            self.pop_size += 1
        self._expl_strat = SymmParamExplStrat(NormalParamNoise(
            self._policy.num_param,
            std_init=expl_std_init,
            std_min=expl_std_min,
        ))

        if optim == 'SGD':
            self.optim = to.optim.SGD([{'params': self._policy.parameters()}], lr=lr, momentum=0.8, dampening=0.1)
        elif optim == 'Adam':
            # self.optim = to.optim.Adam([{'params': self._policy.parameters()}], lr=lr)
            self.optim = to.optim.Adam([{'params': self._distribution.get_params()}], lr=lr)
        else:
            raise NotImplementedError

        self._iter = 0

    def _optimize_distribution_parameters(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if not self._real_env:
            # Sample new policy parameters
            paramsets = self._expl_strat.sample_param_sets(
                self._policy.param_values,
                num_samples=10,
                # If you do not want to include the current policy parameters, be aware that you also have to do follow-up
                # changes in the update() functions in all subclasses of ParameterExploring
                # include_nominal_params=True
                include_nominal_params = True
            )

            with to.no_grad():
                # Sample rollouts using these parameters
                param_samp_res = self.sampler.sample(paramsets)

            # Evaluate the current policy (first one in list if include_nominal_params is True)
            ret_avg_curr = param_samp_res[0].mean_undiscounted_return

            # Store the average return for the stopping criterion
            self.ret_avg_stack = np.delete(self.ret_avg_stack, 0)
            self.ret_avg_stack = np.append(self.ret_avg_stack, ret_avg_curr)

            all_rets = param_samp_res.mean_returns
            all_lengths = np.array([len(ro) for pss in param_samp_res for ro in pss.rollouts])

            # Log metrics computed from the old policy (before the update)
            self.logger.add_value('curr policy return', ret_avg_curr)
            self.logger.add_value('max return', float(np.max(all_rets)))
            self.logger.add_value('median return', float(np.median(all_rets)))
            self.logger.add_value('avg return', float(np.mean(all_rets)))
            self.logger.add_value('std return', float(np.std(all_rets)))
            self.logger.add_value('avg rollout len', float(np.mean(all_lengths)))
            # self.logger.add_value('min mag policy param',
            #                       self._policy.param_values[to.argmin(abs(self._policy.param_values))])
            # self.logger.add_value('max mag policy param',
            #                       self._policy.param_values[to.argmax(abs(self._policy.param_values))])

            # Logging
            self.logger.add_value('policy param', self._policy.param_values.detach().numpy())
            self.logger.add_value('expl strat mean', deepcopy(self._distribution.get_mean(tensor=False)))
            self.logger.add_value('expl strat cov', deepcopy(self._distribution.get_cov(tensor=False)))

            # Extract the best policy parameter sample for saving it later
            self.best_policy_param = param_samp_res.parameters[np.argmax(param_samp_res.mean_returns)].clone()

            # Save snapshot data
            self.make_snapshot(snapshot_mode, float(np.max(param_samp_res.mean_returns)), meta_info)
            # Update the policy
            self.update(param_samp_res)


        else:
            input("Change video and Press ENTER")
            print('policy param: ', self._policy.param_values.detach().numpy())
            print('expl strat mean: ', self._distribution.get_mean(tensor=False))
            print('expl strat std: ', np.sqrt(self._distribution.get_cov(tensor=False)))

            ro = rollout(self._env, self.policy, eval=True, render_mode=RenderMode(text=False))

            r =  np.sum(ro.rewards)
            print('curr policy return: ', r)
            # Update the policy
            self.update()

    def policy_return(self, Ks):
        with to.no_grad():
            r_l = []
            for i, K in enumerate(Ks):
                print("{}/{} - K: {}".format(i, len(K)-1, K.view(-1)), end=" ")

                self.policy.param_values = K
                ro = rollout(self._env, self.policy, eval=True, render_mode=RenderMode(text=False))
                r = np.sum(ro.rewards)
                r_l.append(r)
                print(" - r: {}".format(r))
            return r_l

    def update(self, param_results: ParameterSamplingResult = None, ret_avg_curr: float = None):

        loss = -self._mvd_gaussian_diag_covariance_surrogate_loss().mean()

        self._optimize_distribution_parameters(loss)

        # Update the policy parameters to the mean of the search distribution
        self._policy.param_values = self._distribution.get_mean(tensor=True).view(-1)


    def _mvd_gaussian_diag_covariance_surrogate_loss(self):
        """
        Builds the loss function for gradient computation with measure value derivatives.
        The gradient is taken wrt the distributional parameters (mean and covariance) of a
        Multivariate Gaussian with Diagonal Covariance.
        """
        mean, std = self._distribution.get_mean_and_std()
        diag_std = std

        dist_samples = self._distribution.sample((self._n_mc_samples_gradient, ))

        # Compute gradient wrt mean
        grad_mean = self._mvd_grad_mean_gaussian_diagonal_covariance(dist_samples)

        # Compute gradient wrt std
        grad_cov = self._mvd_grad_covariance_gaussian_diagonal_covariance(dist_samples)

        # Construct the surrogate loss.
        # Here we still backpropagate through the mean and covariance, because they can themselves be parametrized
        surrogate_loss = grad_mean.detach() * mean
        surrogate_loss += grad_cov.detach() * diag_std

        # The total derivative is the sum of the partial derivatives wrt each parameter.
        loss = surrogate_loss.sum(dim=-1)

        return loss

    def _mvd_grad_mean_gaussian_diagonal_covariance(self, dist_samples):
        """
        Computes the measure valued gradient wrt the mean of the multivariate Gaussian with diagonal Covariance.
        """
        print("----Grad mean", flush=True)

        mean, std = self._distribution.get_mean_and_std()
        diag_std = std

        # Replicate the second to last dimension
        # (B, D, D)
        multiples = [1, self._dims, 1]
        base_samples = to.unsqueeze(dist_samples, -2).repeat(*multiples)

        # Sample (B, D) samples from the positive and negative Univariate Weibull distributions
        weibull = torchdist.weibull.Weibull(scale=np.sqrt(2.), concentration=2.)
        pos_samples_weibull = weibull.sample(dist_samples.shape)

        if self._coupling:
            neg_samples_weibull = pos_samples_weibull
        else:
            neg_samples_weibull = weibull.sample(dist_samples.shape)

        # Build the (B, D) positive and negative diagonals of the MVD decomposition
        positive_diag = mean + diag_std * pos_samples_weibull
        assert positive_diag.shape == dist_samples.shape

        negative_diag = mean - diag_std * neg_samples_weibull
        assert negative_diag.shape == dist_samples.shape

        # Set the positive and negative points where to evaluate the Q function.
        # (B, D, D)
        # Replace the ith dimension of the actions with the ith entry of the constructed diagonals.
        # Mohamed. S, 2019, Monte Carlo Gradient Estimation in Machine Learning, Ch. 6.2
        positive_samples = base_samples.clone()
        positive_samples.diagonal(dim1=-2, dim2=-1).copy_(positive_diag)
        negative_samples = base_samples.clone()
        negative_samples.diagonal(dim1=-2, dim2=-1).copy_(negative_diag)

        # MVD constant term
        # (B, D)
        c = np.sqrt(2 * np.pi) * diag_std

        # Evaluate the function
        # pos_f_samples = self._func.eval(positive_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims))
        # neg_f_samples = self._func.eval(negative_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims))

        if not self._real_env:
            pos_paramsets = positive_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims)
            pos_f_samples_param_samp_res = self.sampler.sample(pos_paramsets.detach())
            r_l = []
            for i in range(len(pos_f_samples_param_samp_res)):
                r_l.append(pos_f_samples_param_samp_res[i].mean_undiscounted_return)
            pos_f_samples = to.tensor(r_l)


            neg_paramsets = negative_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims)
            neg_f_samples_param_samp_res = self.sampler.sample(neg_paramsets.detach())
            r_l = []
            for i in range(len(neg_f_samples_param_samp_res)):
                r_l.append(neg_f_samples_param_samp_res[i].mean_undiscounted_return)
            neg_f_samples = to.tensor(r_l)
        else:
            pos_f_samples = to.tensor(self.policy_return(positive_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims)))
            neg_f_samples = to.tensor(self.policy_return(negative_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims)))

        # Gradient batch
        # (B, D)
        delta_f = pos_f_samples - neg_f_samples
        grad = delta_f.reshape(dist_samples.shape[0], self._dims) / c
        assert grad.shape == dist_samples.shape

        return grad

    def _mvd_grad_covariance_gaussian_diagonal_covariance(self, dist_samples):
        """
        Computes the measure valued gradient wrt the covariance of the multivariate Gaussian with diagonal covariance.
        """
        print("----Grad covariance", flush=True)

        mean, std = self._distribution.get_mean_and_std()
        diag_std = std

        # Replicate the second to last dimension of actions
        # (B, D, D)
        multiples = [1, self._dims, 1]
        base_actions = to.unsqueeze(dist_samples, -2).repeat(*multiples)

        # Sample (NxBxDa, Da) samples from the positive and negative Univariate distributions of the decomposition.
        # The positive part is a Double-sided Maxwell M(mu, sigma^2).
        #   M(x; mu, sigma^2) = 1/(sigma*sqrt(2*pi)) * ((x-mu)/sigma)^2 * exp(-1/2*((x-mu)/sigma)^2)
        #   To sample Y from the Double-sided Maxwell M(mu, sigma^2) we can do
        #   X ~ M(0, 1) -> Y = mu + sigma * X
        # The negative part is a Gaussian distribution N(mu, sigma^2).
        #   To sample Y from the Gaussian N(mu, sigma^2) we can do
        #   X ~ N(0, 1) -> Y = mu + sigma * X
        double_sided_maxwell_standard = DoubleSidedStandardMaxwell()
        pos_samples_double_sided_maxwell_standard = double_sided_maxwell_standard.sample(dist_samples.shape)

        if self._coupling:
            # Construct standard Gaussian samples from standard Double-sided Maxwell samples
            neg_samples_gaussian_standard = std_gaussian_from_std_dsmaxwell(pos_samples_double_sided_maxwell_standard)
        else:
            gaussian_standard = torchdist.normal.Normal(loc=0., scale=1.)
            neg_samples_gaussian_standard = gaussian_standard.sample(dist_samples.shape)

        pos_samples_double_sided_maxwell_standard = pos_samples_double_sided_maxwell_standard

        # Build the (B, D) positive and negative diagonals of the MVD decomposition
        positive_diag = mean + diag_std * pos_samples_double_sided_maxwell_standard
        assert positive_diag.shape == dist_samples.shape

        negative_diag = mean + diag_std * neg_samples_gaussian_standard
        assert negative_diag.shape == dist_samples.shape

        # Set the positive and negative points where to evaluate the Q function.
        # (B, D, D)
        # In multivariate Gaussians with diagonal covariance, the univariates are independent.
        # Hence we can replace the ith dimension of the sampled actions with the ith entry of the constructed diagonals.
        # Mohamed. S, 2019, Monte Carlo Gradient Estimation in Machine Learning, Ch. 6.2
        positive_samples = base_actions.clone()
        positive_samples.diagonal(dim1=-2, dim2=-1).copy_(positive_diag)
        negative_samples = base_actions.clone()
        negative_samples.diagonal(dim1=-2, dim2=-1).copy_(negative_diag)

        # MVD constant term
        # (B, D)
        c = diag_std

        # Evaluate the function
        # pos_f_samples = self._func.eval(positive_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims))
        # neg_f_samples = self._func.eval(negative_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims))

        if not self._real_env:
            pos_paramsets = positive_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims)
            pos_f_samples_param_samp_res = self.sampler.sample(pos_paramsets.detach())
            r_l = []
            for i in range(len(pos_f_samples_param_samp_res)):
                r_l.append(pos_f_samples_param_samp_res[i].mean_undiscounted_return)
            pos_f_samples = to.tensor(r_l)


            neg_paramsets = negative_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims)
            neg_f_samples_param_samp_res = self.sampler.sample(neg_paramsets.detach())
            r_l = []
            for i in range(len(neg_f_samples_param_samp_res)):
                r_l.append(neg_f_samples_param_samp_res[i].mean_undiscounted_return)
            neg_f_samples = to.tensor(r_l)
        else:
            pos_f_samples = to.tensor(self.policy_return(positive_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims)))
            neg_f_samples = to.tensor(self.policy_return(negative_samples.reshape(self._n_mc_samples_gradient * self._dims, self._dims)))


        # Gradient batch
        # (B, D)
        delta_f = pos_f_samples - neg_f_samples
        grad = delta_f.reshape(dist_samples.shape[0], self._dims) / c
        assert grad.shape == dist_samples.shape

        return grad


