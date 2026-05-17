import torch
from tqdm import tqdm
from functools import partial
from typing import Callable


import torch
from torch import Tensor
from dm_ps.bridge import ddim_step, BridgeVE, bridge_stats
from dm_ps.models.edm_wrapper import EDMWrapper

from dm_ps.inv_prob.create_inv_prob_openmrg import InverseProblem, plot_field

import matplotlib.pyplot as plt


def mgps(
    initial_noise: torch.Tensor,
    inverse_problem: InverseProblem,
    epsilon_net: EDMWrapper,
    n_gradient_steps: float = 10,
    lr: float = 3e-2,
    t_mid_type: str = "half",
    # --- sampling
    n_steps: int = 64,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    # ---
    display_im: bool = False,
    display_freq: int = 10,
) -> torch.Tensor:

    timesteps = epsilon_net.mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )
    flipped_timesteps = timesteps.flip(0)

    def log_pot(input):
        return inverse_problem.log_pot(epsilon_net.decode(input))

    tmid_fn = get_t_mid_fn(type=t_mid_type, model=epsilon_net)

    t = flipped_timesteps[0]
    x_tprev = t * initial_noise
    pred_x0 = epsilon_net.denoiser_fn(initial_noise, t)

    for idx, (t, t_prev) in tqdm(
        enumerate(zip(flipped_timesteps[:-1], flipped_timesteps[1:]))
    ):

        x_tprev, pred_x0 = mgps_vi_step(
            x_t=x_tprev,
            pred_x0=pred_x0,
            epsilon_net=epsilon_net,
            log_pot=log_pot,
            n_gradient_steps=n_gradient_steps,
            lr=lr,
            t=t,
            t_prev=t_prev,
            tmid_fn=tmid_fn,
        )

        # XXX uncomment to see the evolution of x_0
        if display_im and idx % display_freq == 0:
            x_0 = epsilon_net.denoiser_fn(x_tprev, t_prev)
            plot_field(epsilon_net.decode(x_0[0]))
            plt.show()

    x_0 = epsilon_net.denoiser_fn(x_tprev, t_prev)
    return epsilon_net.decode(x_0)


def mgps_vi_tmid_step(
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    epsilon_net: EDMWrapper,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    lr: float,
    n_gradient_steps: float,
    t: int,
    t_mid: int,
):

    init_mean_tmid, std_prior = bridge_stats(x_t, pred_x0, t, t_mid, eta=1.0)
    logstd_tmid = torch.tensor(std_prior).log() * torch.ones_like(x_t)

    vmean, vlogstd = (
        init_mean_tmid.requires_grad_(),
        logstd_tmid.clone().requires_grad_(),
    )

    pred_x0t = epsilon_net.denoiser_fn(x_t, t)
    mean_tmid, std_tmid = bridge_stats(x_t, pred_x0t, t, t_mid, eta=1.0)
    logstd_tmid = std_tmid.log()

    kl_fn = partial(
        _rb_elbo,
        mean_prior=mean_tmid,
        logstd_prior=logstd_tmid,
        epsilon_net=epsilon_net,
        log_pot=log_pot,
        t=t_mid,
    )

    optim = torch.optim.Adam(params=[vmean, vlogstd], lr=lr)
    for _ in range(n_gradient_steps):
        optim.zero_grad()
        kl_div = kl_fn(vmean, vlogstd)
        kl_div.backward()
        optim.step()

    vmean.detach_(), vlogstd.detach_()
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)

    return vsample


def mgps_vi_step(
    # i: int,
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    epsilon_net: EDMWrapper,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    lr: float,
    n_gradient_steps: float,
    t: int,
    t_prev: int,
    tmid_fn: int,
):
    t_mid = tmid_fn(t_prev)
    t_mid = t_mid if t_mid <= t_prev else t_prev

    vsample_tmid = mgps_vi_tmid_step(
        x_t=x_t,
        pred_x0=pred_x0,
        epsilon_net=epsilon_net,
        log_pot=log_pot,
        n_gradient_steps=n_gradient_steps,
        lr=lr,
        t=t,
        t_mid=t_mid,
    )

    mean, std = BridgeVE(eta=1.0).bridge_midpoint_stats(
        x_t=x_t, x_s=vsample_tmid, sigma_t=t, sigma_t_prev=t_prev, sigma_s=t_mid
    )
    x_tprev = mean + std * torch.randn_like(mean)

    pred_x0 = epsilon_net.denoiser_fn(vsample_tmid, t_mid)

    return x_tprev, pred_x0


def _rb_elbo(
    vmean: torch.Tensor,
    vlogstd: torch.Tensor,
    mean_prior: torch.Tensor,
    logstd_prior: torch.Tensor,
    epsilon_net: EDMWrapper,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    t: int,
):
    """
    'Rao-Blackwellized' elbo using the Gaussian approximation of the backward transition
    """
    kl_prior = kl_mvn(vmean, vlogstd, mean_prior, logstd_prior)
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)

    pred_x0 = epsilon_net.denoiser_fn(vsample, t)
    int_log_pot_est = -log_pot(pred_x0)

    return int_log_pot_est + kl_prior


def kl_mvn(
    v_mean: torch.Tensor,
    v_logstd: torch.Tensor,
    mean: torch.Tensor,
    logstd: torch.Tensor,
):
    # NOTE `logstd` must be of shape (1,) and `v_logstd` of shape v_mea
    assert v_mean.shape == v_logstd.shape

    return 0.5 * (
        -2 * v_logstd.sum()
        + (torch.norm(v_mean - mean) ** 2.0 + (2.0 * v_logstd).exp().sum())
        / (2.0 * logstd).exp()
    )


def get_t_mid_fn(type="uniform", **kwargs):

    if type == "half":

        def fn(t_prev):
            return t_prev / 2

    elif type == "karras":
        model: EDMWrapper = kwargs.get("model")

        def fn(t_prev):
            sigmas = model.mk_sigmas_fn(
                n_steps=32, sigma_min=0.002, sigma_max=t_prev, rho=3
            )
            return sigmas[16]

    else:
        raise ValueError("Unknown type of midpoint selection")

    return fn
