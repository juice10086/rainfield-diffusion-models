from tqdm import tqdm

import torch
from torch import Tensor
from torch.distributions import Categorical

from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.bridge import bridge_stats
from dm_ps.create_inv_prob import InverseProblem, plot_field

import matplotlib.pyplot as plt


def tds(
    initial_noise: Tensor,
    inv_problem: InverseProblem,
    model: EDMWrapper,
    # --- sampling
    n_steps: int = 50,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    # ---
    gamma: float = 1.0,
    n_particles: int = 3,
    tau: float = 1,
    normalize_grad: bool = False,
    # ---
    display_im: bool = True,
    display_freq: int = 10,
):
    x_shape = model.x_shape
    n_samples = initial_noise.shape[0]
    arranged_mapping = torch.arange(n_samples) * n_particles

    obs, std_obs, H_func = inv_problem.obs, inv_problem.obs_std, inv_problem.H_func

    def H_at_decode(input):
        return H_func.H(model.decode(input))

    sigmas = model.mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )
    flipped_sigmas = sigmas.flip(0)

    x_t = torch.randn(n_particles * n_samples, *x_shape)
    for step_idx, (sigma_t, sigma_t_prev) in tqdm(
        enumerate(zip(flipped_sigmas[:-1], flipped_sigmas[1:]))
    ):

        # propose with DPS approximation
        x_t.requires_grad_()

        x_0 = model.denoiser_fn(x_t, sigma_t)
        # NOTE here, grad norm is used as, grad (|| input ||) = grad (input) / || input ||
        # which equivalent to taking a normalized gradient in DPS
        loss = -torch.norm(obs - H_at_decode(x_0))

        (grad,) = torch.autograd.grad(loss, x_t)
        x_t.detach_()

        # compute weights
        # NOTE here std of the proposal and the prior (prior here has nothing to do with inv prob)
        # are assumed to be the same, if the stds are different (e.g. the usage of different eta)
        # then do account for it when computing log probs
        with torch.no_grad():
            if normalize_grad:
                grad = grad / (grad**2).sum(dim=(-3, -2, -1), keepdim=True).sqrt()

            # propose
            mean_prior, std_prior = bridge_stats(
                x_t, x_0, sigma_t, sigma_t_prev, eta=1.0
            )
            mean_y = mean_prior + gamma * grad
            x_t_prev = mean_y + std_prior * torch.randn_like(x_t)

            # compute weights
            log_proposal = (
                (-0.5 * ((x_t_prev - mean_y) / std_prior) ** 2)
                .view(n_particles * n_samples, -1)
                .sum(dim=-1)
            )
            log_prior = (
                (-0.5 * ((x_t_prev - mean_prior) / std_prior) ** 2)
                .view(n_particles * n_samples, -1)
                .sum(dim=-1)
            )

            std_y = (std_obs**2 + tau * sigma_t_prev**2).sqrt()
            x_0t_prev = model.denoiser_fn(x_t_prev, sigma_t_prev)
            llh = (
                (-0.5 * ((obs - H_at_decode(x_0t_prev)) / std_y) ** 2)
                .view(n_particles * n_samples, -1)
                .sum(dim=-1)
            )

            log_weights = log_prior + llh - log_proposal

        # resamples
        log_weights = log_weights.reshape(n_samples, n_particles)
        resampler = Categorical(logits=log_weights)

        idx_resampling = resampler.sample((n_particles,))
        idx_resampling = idx_resampling.permute(1, 0).reshape(-1)
        # NOTE shift the idx accordingly
        # to account for that fact that indices are between 0 and n_particles-1
        idx_resampling += arranged_mapping.repeat_interleave(n_particles, dim=0)

        x_t = x_t_prev[idx_resampling]

        # XXX uncomment to see the evolution of x_0
        if display_im and step_idx % display_freq == 0:
            samples = model.denoiser_fn(x_t[[0]], sigma_t_prev)
            plot_field(model.decode(samples))
            plt.show()

    x_t = x_t[::n_particles]
    x_0 = model.denoiser_fn(x_t, sigma_t_prev)
    return model.decode(x_0)
