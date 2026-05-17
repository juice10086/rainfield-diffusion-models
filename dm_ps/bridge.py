from typing import Tuple
from tqdm import tqdm

import torch
from torch import Tensor


def bridge_stats(x_t, x_0, sigma_t, sigma_t_prev, eta) -> Tuple[Tensor, Tensor]:

    std = eta * (sigma_t_prev * (sigma_t**2.0 - sigma_t_prev**2.0).sqrt() / sigma_t)
    coeff_residue = (sigma_t_prev**2 - std**2).sqrt() / sigma_t
    mean = x_0 + coeff_residue * (x_t - x_0)

    return mean, std


class BridgeVE:
    def __init__(self, eta: float = 1.0):
        self.eta = eta

    def sample(self, x_t, x_s, sigma_t, sigma_t_prev, sigma_s):
        mean, std = self.bridge_midpoint_stats(x_t, x_s, sigma_t, sigma_t_prev, sigma_s)

        return mean + std * torch.randn_like(x_t)

    def bridge_midpoint_stats(self, x_t, x_s, sigma_t, sigma_t_prev, sigma_s):
        """sigma_s < sigma_t_prev < sigma_t"""
        eta = self.eta

        std = (
            eta
            * (
                (sigma_t_prev**2 - sigma_s**2)
                * (sigma_t**2 - sigma_t_prev**2)
                / (sigma_t**2 - sigma_s**2)
            ).sqrt()
        )

        coeff_residue = (
            (sigma_t_prev**2 - sigma_s**2 - std**2) / ((sigma_t**2 - sigma_s**2))
        ).sqrt()

        mean = x_s + coeff_residue * (x_t - x_s)

        return mean, std

    def bridge_stats(self, x_t, x_0, sigma_t, sigma_t_prev):
        eta = self.eta

        std = eta * (sigma_t_prev * (sigma_t**2.0 - sigma_t_prev**2.0).sqrt() / sigma_t)
        coeff_residue = (sigma_t_prev**2 - std**2).sqrt() / sigma_t
        mean = x_0 + coeff_residue * (x_t - x_0)

        return mean, std


def ddim_step(x_t, denoiser_fn, sigma_t, sigma_t_prev, eta):
    pred_x0 = denoiser_fn(x=x_t, sigma=sigma_t)
    mean, std = bridge_stats(x_t, pred_x0, sigma_t, sigma_t_prev, eta)
    return mean + std * torch.randn_like(mean)


def ddim(
    initial_noise,
    denoiser_fn,
    mk_sigmas_fn,
    n_steps,
    sigma_min=0.0,
    sigma_max=1e4,
    eta=0.0,
    rho=7.0,
    **kwargs,
):
    """
    DDIM for variance exploding.
    """
    device = initial_noise.device

    sigmas = mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    ).to(device)
    sigmas_flipped = sigmas.flip(0)

    pbar = tqdm(zip(sigmas_flipped[:-1], sigmas_flipped[1:]))
    x_t = sigmas_flipped[0] * initial_noise
    for sigma_t, sigma_t_prev in pbar:
        x_t = ddim_step(x_t, denoiser_fn, sigma_t, sigma_t_prev, eta)

    return denoiser_fn(x_t, sigma_t_prev)
