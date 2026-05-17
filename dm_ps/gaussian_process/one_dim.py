import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

import matplotlib.pyplot as plt

dtype = torch.float64
torch.set_default_dtype(dtype)


class GaussianProcess1D:
    """Gaussian Process with zero mean and RBF kernel.

    GP(0, k) where k::

        k(t, t') = std^2  exp(-| t - t'|^2 / (2 \ell^2))
    """

    def __init__(self, characteristic_length: float, std: float = 1.0):
        self.characteristics_length = characteristic_length
        self.std = std

    def eval_gp(self, t_range):
        # return (mean, cov): the GP evaluated on t_range
        return MultivariateNormal(self.mean(t_range), self.cov(t_range))

    def mean(self, t_range):
        return torch.zeros_like(t_range)

    def cov(self, t_range):
        return self._rbf_cov(t_range)

    def _rbf_cov(self, t_range: Tensor):
        # NOTE t_range of shape (n_samples,)
        ell = self.characteristics_length
        log_probs = -0.5 * ((t_range[:, None] - t_range) / ell) ** 2
        return self.std**2 * log_probs.exp()


class PosteriorGaussianProcess1DLineInt(GaussianProcess1D):
    """Posterior of Gaussian Process.

    X is a GP(0, k), zero mean and RBF kernel k.
    For intervals [(a_i, b_i)], the inverse problem defined as::

        y_i = \int_{a_i}^{b_i} X(t) dt + std_obs * Z

    where Z is standard Gaussian.
    """

    def __init__(
        self,
        characteristics_length: float,
        std: float = 1.0,
        intervals: Tensor = None,
        obs: Tensor = None,
        std_obs: Tensor = None,
    ):
        # defines the prior
        super().__init__(characteristics_length, std)

        # inverse problem
        self.intervals = intervals
        self.std_obs = std_obs
        self.obs = obs

        # def constant
        self.sqrt_2 = torch.tensor(2.0).sqrt()
        self.sqrt_pi_2 = (torch.tensor(torch.pi) / 2).sqrt()

        # cache computation
        self.cov_obs = self._compute_cov_obs(std_obs)
        self.inv_cov_obs = self.cov_obs.inverse()

    def mean(self, t_range: Tensor) -> Tensor:
        obs, inv_cov_obs = self.obs, self.inv_cov_obs

        # computes self._cross_cov(t_range) @ (inv_cov_obs @ obs)
        return torch.einsum("bi,ij,j->b", self._cross_cov(t_range), inv_cov_obs, obs)

    def cov(self, t_range: Tensor) -> Tensor:
        inv_cov_obs = self.inv_cov_obs

        cov_prior = super().cov(t_range)
        cross_cov = self._cross_cov(t_range)

        # computes cov_prior - cross_cov @ inv_cov_obs @ cross_cov.T
        return cov_prior - torch.einsum(
            "bi,ij,cj->bc", cross_cov, inv_cov_obs, cross_cov
        )

    def _cross_cov(self, t_range: Tensor) -> Tensor:
        # cov(X, Y)
        std, ell = self.std, self.characteristics_length
        sqrt_2, sqrt_pi_2 = self.sqrt_2, self.sqrt_pi_2

        intervals = self.intervals

        left = (t_range[:, None] - intervals[:, 0]) / (sqrt_2 * ell)
        right = (t_range[:, None] - intervals[:, 1]) / (sqrt_2 * ell)

        return std**2 * ell * sqrt_pi_2 * (torch.erf(left) - torch.erf(right))

    def _compute_cov_obs(self, std_obs) -> Tensor:
        # cov(Y, Y)
        intervals = self.intervals
        std, ell = self.std, self.characteristics_length
        sqrt_2, sqrt_pi_2 = self.sqrt_2, self.sqrt_pi_2

        # shape (n_intervals, 2, 2, n_intervals)
        # NOTE the 2 by 2 sub matrices contain
        #   a_i - a_j   b_i - a_j
        #   a_i - b_j   b_i - b_j
        # NOTE sign doesn't matter as ``_primitive_scaled_erf`` is even function
        paired_diff = intervals[..., None, None] - intervals.T

        cst = std**2 * ell * sqrt_pi_2

        noise_less_cov = cst * (
            self._primitive_scaled_erf(paired_diff[:, 0, 1, :])
            + self._primitive_scaled_erf(paired_diff[:, 1, 0, :])
            - self._primitive_scaled_erf(paired_diff[:, 0, 0, :])
            - self._primitive_scaled_erf(paired_diff[:, 1, 1, :])
        )

        # TODO can be made efficient by avoiding to build identity matrix
        return noise_less_cov + std_obs**2 * torch.eye(len(intervals))

    def _primitive_scaled_erf(self, input: Tensor) -> Tensor:
        std, ell = self.std, self.characteristics_length
        sqrt_2, sqrt_pi_2 = self.sqrt_2, self.sqrt_pi_2

        scale_input = input / (sqrt_2 * ell)
        scale = ell / sqrt_pi_2

        return input * torch.erf(scale_input) + scale * torch.exp(-(scale_input**2))
