from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor

from dm_ps.inv_prob.create_inv_prob_openmrg import InverseProblem
from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.bridge import bridge_stats

from dm_ps.inv_prob.create_inv_prob_openmrg import plot_field

import matplotlib.pyplot as plt


def daps(
    initial_noise: Tensor,
    inv_problem: InverseProblem,
    model: EDMWrapper,
    n_steps: int = 32,
    n_diffusion_steps: int = 2,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    eta_x0_est: float = 0.0,
    tau: float = 1.0,
    mcmc_sampler: str = "Langevin",
    n_mcmc_steps: int = 20,
    lr: float = 1e-6,
    min_ratio: float = 0.43,
    mcmc_sampler_extra_params=None,
    # ---
    display_im: bool = False,
):
    device = initial_noise.device
    dtype = initial_noise.dtype

    if mcmc_sampler_extra_params is None:
        mcmc_sampler_extra_params = {}

    if mcmc_sampler == "Langevin":
        mcmc_step = Langevin(initial_noise.shape)
    elif mcmc_sampler == "HMC":
        momentum = mcmc_sampler_extra_params.get("momentum", 0.9)
        mcmc_step = HMC(momentum=momentum)
    else:
        raise ValueError

    sigmas = model.mk_sigmas_fn(
        n_steps=n_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
    ).to(device=device, dtype=dtype)
    sigmas = sigmas.flip(0)

    x_t = sigmas[0] * initial_noise

    pbar = tqdm(range(len(sigmas) - 1))
    for k in pbar:
        sigma_t = sigmas[k]
        sigma_prev = sigmas[k + 1]

        x0_t = _estimate_x0t_ve(
            x_t=x_t,
            model=model,
            sigma_t=sigma_t,
            sigma_min=float(sigma_min),
            n_diffusion_steps=n_diffusion_steps,
            eta=eta_x0_est,
        )

        x0_i = x0_t.clone()
        # to avoid dividing by zero
        std_prior = sigma_t.clamp_min(1e-12)
        current_lr = _get_lr(k / max(1, (len(sigmas) - 1)), lr, min_ratio)

        for _ in range(n_mcmc_steps):
            score_prior = -(x0_i - x0_t) / (std_prior**2)

            x0_i.requires_grad_(True)
            x_dec = model.decode(x0_i)
            llh = inv_problem.log_pot(x_dec) / (tau**2)

            if llh.ndim != 0:
                llh = llh.sum()

            llh.backward()
            score_llh = x0_i.grad

            with torch.no_grad():
                x0_i = mcmc_step(x0_i, score_prior + score_llh, current_lr)

        x_t = x0_i + sigma_prev * torch.randn_like(x_t)

        # XXX uncomment to see the evolution of x_0
        if display_im and k % 10 == 0:
            x_0 = model.denoiser_fn(x_t, sigma_prev)
            plot_field(model.decode(x_0[0]))
            plt.show()

    x0 = model.denoiser_fn(x=x_t, sigma=sigmas[-1])
    return model.decode(x0)


def _get_lr(ratio, lr, lr_min_ratio):
    p = 1
    multiplier = (1 ** (1 / p) + ratio * (lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
    return multiplier * lr


def _estimate_x0t_ve(
    x_t: Tensor,
    model,
    sigma_t: Tensor,
    sigma_min: float,
    n_diffusion_steps: int,
    eta: float = 1.0,
) -> Tensor:

    device = x_t.device
    dtype = x_t.dtype

    sigmas_short = model.mk_sigmas_fn(
        n_steps=n_diffusion_steps + 1,
        sigma_min=sigma_min,
        sigma_max=sigma_t,
        rho=7,
    ).to(device=device, dtype=dtype)
    sigmas_short = sigmas_short.flip(0)

    x = x_t
    for i in range(len(sigmas_short) - 1):
        s_i = sigmas_short[i]
        s_next = sigmas_short[i + 1]

        pred_x0 = model.denoiser_fn(x=x, sigma=s_i)
        mean, std = bridge_stats(x, pred_x0, s_i, s_next, eta=eta)
        x = mean if eta == 0 else mean + std * torch.randn_like(mean)

    x0 = model.denoiser_fn(x=x, sigma=sigmas_short[-1])
    return x0


class Langevin:
    def __init__(self, x_shape):
        self.x_shape = x_shape

    def __call__(self, x, score, lr):
        noise = torch.randn(*self.x_shape)
        return x + lr * score + np.sqrt(2 * lr) * noise


class HMC:
    def __init__(self, x_shape, momentum):
        self.x_shape = x_shape
        self.momentum = momentum
        self.velocity = torch.randn(x_shape)

    def __call__(self, x_t, score, lr):
        noise = torch.randn(*self.x_shape)

        step_size = np.sqrt(lr)
        self.velocity = (
            self.momentum * self.velocity
            + step_size * score
            + np.sqrt(2 * (1 - self.momentum)) * noise
        )

        return x_t + self.velocity * step_size
