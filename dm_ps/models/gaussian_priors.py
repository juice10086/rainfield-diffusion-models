# %%
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical

from dm_ps.bridge import ddim, ddim_step


class DenoiserGaussianEDM(torch.nn.Module):

    def __init__(self, mean, cov, device="cpu"):
        super().__init__()

        self.mean = mean
        self.cov = cov
        self.device = device

        self.x_shape = (len(mean),)

    def denoiser_fn(self, x: torch.Tensor, sigma, *args, **kwargs):
        score = self.score_fn(x, sigma)

        while sigma.ndim < score.ndim:
            sigma = sigma.unsqueeze(-1)

        return x + sigma**2 * score

    def score_fn(self, x: Tensor, sigma, *args, **kwargs):

        grad_fn = torch.func.grad(
            lambda input: self.log_marginal(input, sigma).sum(),
        )

        return grad_fn(x)

    def exact_backward(self, x_t, sigma_t, sigma_t_prev):
        mean, cov = self.mean, self.cov
        dim = mean.shape[0]
        I = torch.eye(dim)

        diff = sigma_t**2 - sigma_t_prev**2
        inv_cov_t_prev = (cov + sigma_t_prev**2 * I).inverse()
        cov_bwd = (1 / diff * I + inv_cov_t_prev).inverse()

        if x_t.ndim == 1:
            mean_bwd = cov_bwd @ (x_t / diff + inv_cov_t_prev @ mean)
        else:
            # first dim is batch
            mean_bwd = torch.einsum(
                "ij,bj->bi", cov_bwd, (x_t / diff + (inv_cov_t_prev @ mean)[None])
            )

        return mean_bwd, cov_bwd

    def log_marginal(self, x_t, sigma) -> Tensor:
        ndim = len(self.mean)

        # HACK to be fixed as it doesn't handle general cases
        sigma = sigma.squeeze(-1)
        cov_noise = torch.einsum("...,ij->...ij", sigma**2, torch.eye(ndim))

        mean_t = self.mean
        while mean_t.ndim < x_t.ndim:
            mean_t = mean_t.unsqueeze(0)

        cov_t = self.cov
        while cov_t.ndim < cov_noise.ndim:
            cov_t = cov_t.unsqueeze(0)

        # cov_t = cov + sigma**2 * I
        # avoid inplace add to not override cov
        cov_t = cov_t + cov_noise

        marginal_t = MultivariateNormal(loc=mean_t, covariance_matrix=cov_t)
        return marginal_t.log_prob(x_t)

    def mk_sigmas_fn(
        self,
        n_steps: int,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    ) -> torch.Tensor:
        sigmas = (
            sigma_max ** (1 / rho)
            + torch.linspace(1, 0, n_steps)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        return sigmas

    def forward(self, x: Tensor, sigma, *args, **kwargs):
        return self.denoiser_fn(x, sigma)

    def pred_noise(self, x: Tensor, sigma, *args, **kwargs):
        x_0 = self.denoiser_fn(x, sigma, *args, **kwargs)
        return (x - x_0) / sigma

    # --- for compatibility with EDM wrapper
    def encode(self, x):
        return x

    def decode(self, x):
        return x


class DenoiserGaussianMixtureEDM(torch.nn.Module):

    def __init__(self, means, covs, weights, device="cpu"):
        super().__init__()

        self.means = means
        self.covs = covs
        self.weights = weights
        self.device = device

        self.x_shape = (len(mean),)

    def denoiser_fn(self, x: torch.Tensor, sigma, *args, **kwargs):
        score = self.score_fn(x, sigma)

        while sigma.ndim < score.ndim:
            sigma = sigma.unsqueeze(-1)

        return x + sigma**2 * score

    def score_fn(self, x: Tensor, sigma, *args, **kwargs):

        grad_fn = torch.func.grad(
            lambda input: self.log_marginal(input, sigma).sum(),
        )

        return grad_fn(x)

    def log_marginal(self, x_t, sigma) -> Tensor:
        ndim = self.means.shape[1]

        # HACK to be fixed as it doesn't handle general cases
        sigma = sigma.squeeze(-1)
        cov_noise = torch.einsum("...,ij->...ij", sigma**2, torch.eye(ndim))

        covs_t = self.covs
        while covs_t.ndim < cov_noise.ndim:
            covs_t = covs_t.unsqueeze(0)

        covs_t = self.covs.unsqueeze(-4) + cov_noise.unsqueeze(-3)

        means_t = self.means
        while means_t.ndim < x_t.ndim + 1:
            means_t = means_t.unsqueeze(0)

        weights = self.weights
        while weights.ndim < means_t.ndim - 1:
            weights = weights.unsqueeze(0)

        marginal_t = MixtureSameFamily(
            mixture_distribution=Categorical(self.weights),
            component_distribution=MultivariateNormal(
                loc=means_t, covariance_matrix=covs_t
            ),
        )
        return marginal_t.log_prob(x_t)

    def mk_sigmas_fn(
        self,
        n_steps: int,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    ) -> torch.Tensor:
        sigmas = (
            sigma_max ** (1 / rho)
            + torch.linspace(1, 0, n_steps)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        return sigmas

    def forward(self, x: Tensor, sigma, *args, **kwargs):
        return self.denoiser_fn(x, sigma)

    def pred_noise(self, x: Tensor, sigma, *args, **kwargs):
        x_0 = self.denoiser_fn(x, sigma, *args, **kwargs)
        return (x - x_0) / sigma


class PosteriorLinearInvProb(MultivariateNormal):
    """Posterior distribution of Linear inverse problem under Gaussian prior.

    Defines ``p(x | obs)``, where::

        obs = A x + obs_std * n
        x ~ N(mean, cov)
    """

    def __init__(self, A, obs, std_obs, mean_prior, covariance_prior):

        mean, cov = mean_prior, covariance_prior
        inv_cov = torch.inverse(cov)

        inv_Sigma = inv_cov + A.T @ A / std_obs**2
        Sigma = inv_Sigma.inverse()

        m = A.T @ obs / std_obs**2 + inv_cov @ mean

        super().__init__(loc=Sigma @ m, covariance_matrix=Sigma)


class PosteriorLinearInvProvGMM(MixtureSameFamily):

    def __init__(self, A, obs, std_obs, means, covs, weights):
        dim = means.shape[-1]

        # --- weights of the posterior
        # for cov in covs: std**2 I + A @ cov @ A.T
        weights_covs = std_obs**2 * torch.eye(dim)
        weights_covs = weights_covs[None] + torch.einsum("ik,ckl,jl->cij", A, covs, A)
        weights_means = torch.einsum("ij,cj->ci", A, means)

        weights_posterior = (
            weights
            * MultivariateNormal(weights_means, weights_covs).log_prob(obs[None]).exp()
        )

        # --- means/covs of the components
        inv_covs = torch.inverse(covs)
        inv_Sigma = inv_covs + (A.T @ A / std_obs**2)[None]
        Sigma = torch.inverse(inv_Sigma)

        m = (A.T @ obs / std_obs**2)[None] + torch.einsum("cij,cj->ci", inv_covs, means)
        m = torch.einsum("cij,cj->ci", Sigma, m)

        super().__init__(
            Categorical(weights_posterior),
            component_distribution=MultivariateNormal(m, Sigma),
        )


def sliced_wasserstein(
    dist_1: torch.Tensor, dist_2: torch.Tensor, n_slices: int = 100
) -> float:
    """Compute sliced Wasserstein distance between two distributions.

    Assumes that both ``dist_1`` and ``dist_2`` have the same dimension.

    Parameters
    ----------
    dist_1 : Tensor

    dist_2 : Tensor

    n_slices : int, default=100
        The number of the considered random projections.

    Return
    ------
    sw_distance : float
    """
    import numpy as np
    from scipy.stats import wasserstein_distance

    if dist_1.ndim > 2:
        dist_1 = dist_1.reshape(dist_1.shape[0], -1)
        dist_2 = dist_2.reshape(dist_2.shape[0], -1)

    projections = torch.randn(size=(n_slices, dist_1.shape[1]), device=dist_1.device)
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T

    dist_1_projected = dist_1_projected.cpu().numpy()
    dist_2_projected = dist_2_projected.cpu().numpy()

    return np.mean(
        [
            wasserstein_distance(u_values=d1, v_values=d2)
            for d1, d2 in zip(dist_1_projected, dist_2_projected)
        ]
    )


# %%
if __name__ == "__main__":
    # example with a single Gaussian
    dim = 2

    mean = torch.zeros(2)
    cov = torch.tensor([[1, 0.5], [0.5, 1]])

    ref_dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
    model = DenoiserGaussianEDM(mean, cov)

    # %%
    # example with Mixture of Gaussian
    dim = 2
    means = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    covs = 0.25 * torch.tensor(
        [
            [[1, 0.5], [0.5, 1]],
            [[1, -0.5], [-0.5, 1]],
        ]
    )
    weights = torch.tensor([0.5, 0.5])

    ref_dist = MixtureSameFamily(
        mixture_distribution=Categorical(weights),
        component_distribution=MultivariateNormal(loc=means, covariance_matrix=covs),
    )
    model = DenoiserGaussianMixtureEDM(means, covs, weights)

    # %%
    # # checking batch of x_t with different noise levels
    # n_samples = 10_000

    # sigmas = model.mk_sigmas_fn(n_steps=100, sigma_min=0.00, sigma_max=100, rho=7)
    # sigmas_flipped = sigmas.flip(0)
    # sigmas_flipped = sigmas_flipped[:, None].repeat((1, n_samples))
    # sigmas_flipped = sigmas_flipped[..., None]

    # initial_noise = torch.randn((n_samples, dim))
    # x_t = sigmas_flipped[0] * initial_noise

    # for sigma_t, sigma_t_prev in zip(sigmas_flipped[:-1], sigmas_flipped[1:]):
    #     x_t = ddim_step(x_t, model.denoiser_fn, sigma_t, sigma_t_prev, eta=1.0)

    # x = x_t
    # %%
    n_samples = 10_000
    initial_noise = torch.randn((n_samples, dim))

    x = ddim(
        initial_noise,
        model.denoiser_fn,
        model.mk_sigmas_fn,
        n_steps=100,
        sigma_min=0.00,
        sigma_max=100,
        rho=7,
    )

    # %%
    import matplotlib.pyplot as plt

    ref_samples = ref_dist.sample((n_samples,))

    fig, ax = plt.subplots()

    opacity = 0.5
    ax.scatter(x[:, 0], x[:, 1], label="generated", alpha=opacity)
    ax.scatter(ref_samples[:, 0], ref_samples[:, 1], label="reference", alpha=opacity)

    ax.set_aspect(1)
    ax.legend()

    # %%
    # using exact backward
    n_samples = 10_000
    initial_noise = torch.randn((n_samples, dim))
    sigmas = model.mk_sigmas_fn(n_steps=50, sigma_max=100, rho=7)

    sigmas_flipped = sigmas.flip(0)
    pbar = zip(sigmas_flipped[:-1], sigmas_flipped[1:])

    x_t = sigmas_flipped[0] * initial_noise
    for sigma_t, sigma_t_prev in pbar:

        mean_bwd, cov_bwd = model.exact_backward(x_t, sigma_t, sigma_t_prev)
        x_t = MultivariateNormal(
            loc=mean_bwd,
            covariance_matrix=cov_bwd[None],
        ).sample(
            (1,)
        )[0]

    ref_samples = ref_dist.sample((n_samples,))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    opacity = 0.5
    ax.scatter(x_t[:, 0], x_t[:, 1], label="generated", alpha=opacity)
    ax.scatter(ref_samples[:, 0], ref_samples[:, 1], label="reference", alpha=opacity)

    ax.set_aspect(1)
    ax.legend()
