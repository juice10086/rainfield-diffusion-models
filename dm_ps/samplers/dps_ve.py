from tqdm import tqdm
from typing import Tuple, Callable

import torch
from torch import Tensor
from dm_ps.bridge import ddim_step, BridgeVE, bridge_stats
from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.create_inv_prob import InverseProblem, plot_field

import matplotlib.pyplot as plt


def dps(
    initial_noise: Tensor,
    inverse_problem: InverseProblem,
    model: EDMWrapper,
    gamma: float = 1.0,
    normalize_grad: bool = False,
    # --- sampling
    n_steps: int = 50,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    # ---
    eta: float = 1.0,
    display_im: bool = False,
    display_freq: int = 10,
) -> Tensor:
    obs, H_func = inverse_problem.obs, inverse_problem.H_func

    timesteps = model.mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )
    flipped_timesteps = timesteps.flip(0)

    x_t = flipped_timesteps[0] * initial_noise
    for idx, (sigma_t, sigma_t_prev) in tqdm(
        enumerate(zip(flipped_timesteps[:-1], flipped_timesteps[1:]))
    ):
        x_t.requires_grad_()
        x_0 = model.denoiser_fn(x_t, sigma_t)
        # NOTE here, grad norm is used as, grad (|| input ||) = grad (input) / || input ||
        # which equivalent to taking a normalized gradient in DPS
        loss = -torch.norm(obs - H_func.H(model.decode(x_0)))

        (grad,) = torch.autograd.grad(loss, x_t)
        x_t.detach_()

        with torch.no_grad():
            if normalize_grad:
                grad = grad / (grad**2).sum(dim=(-3, -2, -1), keepdim=True).sqrt()

            mean, std = bridge_stats(x_t, x_0, sigma_t, sigma_t_prev, eta=eta)
            x_t = mean + gamma * grad + std * torch.randn_like(x_t)

        # XXX uncomment to see the evolution of x_0
        if display_im and idx % display_freq == 0:
            sample = model.denoiser_fn(x_t[[0]], sigma_t_prev)
            plot_field(model.decode(sample))
            plt.show()

    x_0 = model.denoiser_fn(x_t, sigma_t_prev)
    return model.decode(x_0)
