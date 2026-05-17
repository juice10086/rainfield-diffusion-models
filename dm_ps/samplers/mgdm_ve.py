from tqdm import tqdm
from typing import Tuple, Callable

import torch
from torch import Tensor
from dm_ps.bridge import ddim_step, BridgeVE
from dm_ps.models.edm_wrapper import EDMWrapper

from dm_ps.inv_prob.create_inv_prob_openmrg import InverseProblem, plot_field

import matplotlib.pyplot as plt


def mgdm(
    initial_noise: torch.Tensor,
    inverse_problem: InverseProblem,
    epsilon_net: EDMWrapper,
    n_reps: int,
    lr: float = 3e-2,
    n_gradient_steps: int = 10,
    n_denoising_steps: int = 20,
    tau_sampling: int = "uniform",
    min_tau: int = 2e-3,
    # --- sampling
    n_steps: int = 50,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    # ---
    eta: float = 1.0,
    display_im: bool = True,
) -> Tensor:

    log_pot = inverse_problem.log_pot
    timesteps = epsilon_net.mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )

    lr_fn = lambda _: lr
    grad_steps_fn = lambda _: n_gradient_steps
    tau_sample_fn = get_tau_sampling(
        type=tau_sampling, min_tau=min_tau, model=epsilon_net
    )

    bridge = BridgeVE(eta=eta)

    x_t = timesteps[-1] * initial_noise
    x_0 = epsilon_net.denoiser_fn(x_t, timesteps[-1])

    flipd_timesteps = timesteps.flip(0)
    pbar = tqdm(
        enumerate(
            zip(flipd_timesteps[2:-1], flipd_timesteps[1:-2], flipd_timesteps[:-2])
        )
    )

    for idx, (t_prev, t, t_next) in pbar:

        tau = tau_sample_fn(t_prev, idx)
        ngrad_steps = grad_steps_fn(idx)
        lr = lr_fn(idx)

        x_tnext = x_t
        x_t = bridge.sample(x_tnext, x_0, t_next, t, 0)

        log_pot_fn = lambda x: log_pot(
            epsilon_net.decode(epsilon_net.denoiser_fn(x, tau))
        )

        for _ in range(n_reps):
            x_0, x_t = gibbs_step(
                # fmt: off
                x_0, x_t, x_tnext,
                tau, t, t_next,
                # fmt: on
                epsilon_net,
                log_pot_fn,
                lr,
                ngrad_steps,
                n_denoising_steps,
                eta=eta,
            )

        # XXX uncomment to see the evolution of x_0
        if display_im and idx % 10 == 0:
            plot_field(epsilon_net.decode(x_0[0]))
            plt.show()

    return epsilon_net.decode(x_0)


def gibbs_step(
    # fmt: off
    x_0, x_t, x_tnext,
    tau, t, t_next,
    # fmt: on
    epsilon_net: EDMWrapper,
    log_pot: Callable[[Tensor], Tensor],
    lr: float,
    n_gradient_steps: int,
    n_denoising_steps: int,
    eta: float,
) -> Tuple[Tensor, Tensor]:

    # potential
    x_tau = _pot_update_stochastic(
        # fmt: off
        x_0, x_t,
        tau, t,
        # fmt: on
        log_pot,
        lr,
        n_gradient_steps,
    )

    # denoising step
    sigma_min = 2e-3  # NOTE EDM doesn't allow for sigma_min to be 0 (it returns nan)
    x_0 = _denoising_step(
        sigma_min, tau, x_tau, epsilon_net, n_denoising_steps, eta=eta
    )

    # noising step
    x_t = BridgeVE(eta=eta).sample(x_tnext, x_tau, t_next, t, tau)

    return x_0, x_t


# ------ utils
def _pot_update_stochastic(
    # fmt: off
    x_0, x_t,
    tau, t,
    # fmt: on
    log_pot: Callable[[Tensor], Tensor],
    lr: float,
    n_gradient_steps: int,
) -> Tensor:
    """
    Return
    ------
    x_tau
    """
    mean_prior, std_prior = BridgeVE(eta=1).bridge_midpoint_stats(x_t, x_0, t, tau, 0)
    log_std_prior = std_prior.log()

    vmean = mean_prior.clone()
    vlog_std = log_std_prior * torch.ones_like(x_t)
    vmean.requires_grad_(), vlog_std.requires_grad_()
    optim = torch.optim.Adam(params=[vmean, vlog_std], lr=lr)

    for _ in range(n_gradient_steps):
        optim.zero_grad()
        vsample = vmean + vlog_std.exp() * torch.randn_like(mean_prior)

        loss = (
            -log_pot(vsample)
            + 0.5 * (((vsample - mean_prior) / std_prior) ** 2).sum()
            - vlog_std.sum()
        )
        loss.backward()

        optim.step()

    with torch.no_grad():
        x_tau = vmean + vlog_std.exp() * torch.randn_like(x_t)

    return x_tau


def _denoising_step(
    s, tau, x_tau: Tensor, epsilon_net: EDMWrapper, n_denoising_steps: int, eta: float
):
    sigmas = epsilon_net.mk_sigmas_fn(n_denoising_steps, sigma_min=s, sigma_max=tau)
    sigmas_flipped = sigmas.flip(0)

    pbar = zip(sigmas_flipped[:-1], sigmas_flipped[1:])
    x_t = x_tau
    for sigma_t, sigma_t_prev in pbar:
        x_t = ddim_step(x_t, epsilon_net.denoiser_fn, sigma_t, sigma_t_prev, eta)

    return epsilon_net.denoiser_fn(x_t, sigma_t_prev)


def get_tau_sampling(type="uniform", **kwargs):
    min_tau = kwargs.get("min_tau")

    if type == "uniform":

        def fn(t_prev, idx):
            dist = torch.distributions.Uniform(
                low=min_tau,
                high=t_prev,
            )
            return dist.sample((1,))[0]

    elif type == "karras":
        model: EDMWrapper = kwargs.get("model")

        def fn(t_prev, idx):
            sigmas = model.mk_sigmas_fn(
                n_steps=16, sigma_min=0.002, sigma_max=t_prev, rho=5
            )
            i = torch.randint(low=0, high=16, size=(1,))[0]
            return sigmas[i]

    else:
        raise ValueError("Unknown type of sampling tau")

    return fn
