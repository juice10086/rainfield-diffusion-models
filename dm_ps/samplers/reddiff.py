from tqdm import tqdm

import torch
from torch import Tensor

from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.inv_prob.create_inv_prob_openmrg import InverseProblem, plot_field


import matplotlib.pyplot as plt


def reddiff(
    initial_noise: Tensor,
    inv_problem: InverseProblem,
    model: EDMWrapper,
    # ----
    n_steps: int,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    # ---
    lr: float = 3e-2,
    sigma_x0: float = 1e-3,  # jitter on x0_pred
    grad_term_weight: float = 0.25,
    obs_weight: float = 1.0,
    denoise_weight_mode: str = "linear",  # {'linear','sqrt','square','log','trunc_linear','power2over3','const'}
    # ---
    display_im: bool = False,
    display_freq: int = 10,
):
    obs, H_fn = inv_problem.obs, inv_problem.H_func
    n_samples = initial_noise.shape[0]

    # setup of the model
    timesteps = model.mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )
    flipped_sigmas = timesteps.flip(0)

    # it is not obvious how to choose mu init
    mu = torch.ones_like(initial_noise)
    mu.requires_grad_()

    opt = torch.optim.Adam([mu], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    for step_idx, sigma_t in tqdm(enumerate(flipped_sigmas)):
        opt.zero_grad()

        alpha_t = 1.0

        noise_x0 = torch.randn_like(mu)
        noise_xt = torch.randn_like(mu)

        x0_pred = mu + sigma_x0 * noise_x0
        x_t = alpha_t * x0_pred + sigma_t * noise_xt

        with torch.no_grad():
            eps_pred = model.pred_noise(x_t, sigma_t)

        snr_inv = sigma_t
        w_scale = _get_w_scale(snr_inv, denoise_weight_mode)

        w_t = grad_term_weight * w_scale
        v_t = obs_weight

        # ----- losses
        loss_noise = (
            ((eps_pred - noise_xt).detach() * x0_pred)
            .view(n_samples, -1)
            .mean(dim=-1)
            .sum()
        )

        # Data term
        x_decoded = model.decode(x0_pred)
        Hx = H_fn.H(x_decoded)
        e_obs = obs - Hx
        loss_obs = (0.5 * (e_obs**2)).view(n_samples, -1).mean(dim=-1).sum()

        loss = (w_t * loss_noise) + (v_t * loss_obs)

        loss.backward()
        opt.step()

        # XXX uncomment to see the evolution of x_0
        with torch.no_grad():
            if display_im and step_idx % display_freq == 0:
                plot_field(model.decode(mu[0]))
                plt.show()

    mu.detach_()
    x_final = model.decode(mu)
    return x_final


def _get_w_scale(snr_inv, denoise_weight_mode: str):

    if denoise_weight_mode == "linear":
        w_scale = snr_inv
    elif denoise_weight_mode == "sqrt":
        w_scale = snr_inv.sqrt()
    elif denoise_weight_mode == "square":
        w_scale = snr_inv**2
    elif denoise_weight_mode == "log":
        w_scale = (snr_inv + 1.0).log()
    elif denoise_weight_mode == "trunc_linear":
        w_scale = snr_inv.clamp_max(1.0)
    elif denoise_weight_mode == "power2over3":
        w_scale = snr_inv ** (2.0 / 3.0)
    elif denoise_weight_mode == "const":
        w_scale = torch.ones_like(snr_inv)
    else:
        w_scale = snr_inv

    return w_scale
