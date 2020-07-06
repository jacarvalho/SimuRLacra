import torch
import numpy as np


class GaussianDiagonal:

    def __init__(self, init_loc=(10., 10.), init_std=(1., 1.), device=None):

        self._dims = len(init_loc)

        self._init_loc = init_loc
        self._init_std = init_std

        self._device = torch.device('cpu' if device == None else device)

        self._loc = torch.tensor(self._init_loc, requires_grad=True, device=self._device)
        self._std = torch.tensor(self._init_std, requires_grad=True, device=self._device)

    def sample(self, batch_shape, tensor=True):
        dist = torch.distributions.MultivariateNormal(loc=self._loc, covariance_matrix=torch.diag(torch.pow(self._std, 2)))
        if tensor:
            return dist.rsample(batch_shape)
        return dist.sample(batch_shape).detach().cpu().numpy()

    def get_number_of_dims(self):
        return self._dims

    def get_mean(self, tensor=True):
        if tensor:
            return self._loc.view(1, -1)
        else:
            return self._loc.detach().cpu().numpy().reshape((1, -1))

    def set_mean(self, mu):
        self._loc.data = mu

    def get_std(self, tensor=True):
        if tensor:
            return self._std.view(1, -1)
        else:
            return self._std.detach().cpu().numpy().reshape((1, -1))

    def set_std(self, std):
        self._std.data = std

    def get_cov(self, tensor=True):
        if tensor:
            return torch.pow(self._std, 2)
        else:
            return torch.pow(self._std, 2).view(1, -1).detach().cpu().numpy().reshape((1, -1))

    def get_mean_and_std(self, tensor=True):
        return self.get_mean(tensor=tensor), self.get_std(tensor=tensor)

    def get_params(self):
        return [self._loc, self._std]

    def set_params(self, params):
        loc = params[:self._dims]
        std = params[self._dims:]
        self.set_mean(loc)
        self.set_std(std)

    def log_prob(self, x, tensor=True):
        dist = torch.distributions.MultivariateNormal(loc=self._loc, covariance_matrix=torch.diag(torch.pow(self._std, 2)))
        return dist.log_prob(x)

    def get_grad_mean(self, tensor=True):
        if tensor:
            return self._loc.grad
        else:
            return self._loc.grad.cpu().numpy()

    def get_grad_std(self, tensor=True):
        if tensor:
            return self._std.grad
        else:
            return self._std.grad.cpu().numpy()


class GaussianDiagonalLogStdParametrization:

    def __init__(self, init_loc=(10., 10.), init_std=(1., 1.), device=None):

        self._dims = len(init_loc)

        self._init_loc = init_loc
        self._init_log_std = np.log(init_std, dtype=np.float64)

        self._loc = torch.tensor(self._init_loc, requires_grad=True, device=device, dtype=torch.float64)
        self._log_std = torch.tensor(self._init_log_std, requires_grad=True, device=device, dtype=torch.float64)
        self._cov = torch.exp(2 * self._log_std)

        self._dist = torch.distributions.MultivariateNormal(loc=self._loc, covariance_matrix=torch.diag(self._cov))

    def sample(self, batch_shape, tensor=True):
        cov = torch.diag(torch.exp(2 * self._log_std))
        dist = torch.distributions.MultivariateNormal(loc=self._loc, covariance_matrix=cov)
        if tensor:
            return dist.rsample(batch_shape)
        return dist.sample(batch_shape).detach().cpu().numpy()

    def get_number_of_dims(self):
        return self._dims

    def get_mean(self, tensor=True):
        if tensor:
            return self._loc.view(1, -1)
        else:
            return self._loc.detach().cpu().numpy().reshape((1, -1))

    def set_mean(self, mu):
        self._loc.data = mu

    def get_log_std(self, tensor=True):
        if tensor:
            return self._log_std.view(1, -1)
        else:
            return self._log_std.detach().cpu().numpy().reshape((1, -1))

    def set_log_std(self, log_std):
        self._log_std.data = log_std

    def get_cov(self, tensor=True):
        if tensor:
            return torch.exp(2 * self._log_std).view(1, -1)
        else:
            return torch.exp(2 * self._log_std).detach().cpu().numpy().reshape((1, -1))

    def get_std(self, tensor=True):
        return torch.exp(self._log_std)

    def get_mean_and_std(self, tensor=True):
        return self.get_mean(tensor=tensor), self.get_std(tensor=tensor)

    def get_params(self):
        return [self._loc, self._log_std]

    def set_params(self, params):
        loc = params[:self._dims]
        log_std = params[self._dims:]
        self.set_mean(loc)
        self.set_log_std(log_std)

    def log_prob(self, x, tensor=True):
        mean = self.get_mean()
        cov = torch.diag(self.get_cov().reshape(-1))
        dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
        return dist.log_prob(x)

    def get_grad_mean(self, tensor=True):
        if tensor:
            return self._loc.grad
        else:
            return self._loc.grad.cpu().numpy()

    def get_grad_log_std(self, tensor=True):
        if tensor:
            return self._log_std.grad
        else:
            return self._log_std.grad.cpu().numpy()
