from tqdm import tqdm, trange
import torch
from torch import Tensor
from torch.distributions import Normal, Independent

from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.bridge import ddim_step, bridge_stats
from dm_ps.samplers.replica_exhange.utils import (
    _get_denoised,
    _get_adjacents,
    _update_adjacents,
)

from dm_ps.create_inv_prob import InverseProblem, plot_field

import matplotlib.pyplot as plt


def replica_algo(
    initial_noise: Tensor,
    inverse_problem: InverseProblem,
    model: EDMWrapper,
    n_steps: int = 32,  # length of the path
    n_PT_steps: int = 100,  # total number of Parallel Tempering steps
    n_steps_proposal: int = 1,  # number of steps between adjacent [t, t']
    proposal_kernel: str = "prior",  # kernel used for proposing new state (t, t')
    params_scheduler=None,
    display_im: bool = False,
    display_freq: int = 5,
    return_iterations: bool = False,
    return_only_reconstructions: bool = False,
):
    # NOTE initial_noise has (batch, *dim)
    proposal_kernel = proposal_kernel.lower()
    # NOTE use log_pot_batch as we are interested in log pot of each sample
    log_pot = inverse_problem.log_pot_batch

    # from sigmas[0] = sigma_min ---> sigmas[-1] = sigma_max
    sigmas = model.mk_sigmas_fn(n_steps=n_steps, **params_scheduler)

    # init path x ---> path_x of shape (n_steps, batch, *dim)
    path_x = _init_path(initial_noise, sigmas, model)

    # broadcast ``sigmas`` from shape (n_steps,) ---> (n_steps, 1, 1, ...)
    # for compatibility of operations
    while sigmas.ndim < path_x.ndim:
        sigmas = sigmas.unsqueeze(-1)

    # use dps pots to define path
    def log_pot_sigma(input_x, input_sigs):
        denoised = _get_denoised(input_x, input_sigs, model)
        return log_pot(denoised)

    selected_proposal = AVAILABLE_PROPOSALS[proposal_kernel]

    # to store iterations and acceptance
    iterations = [path_x[0].clone()[None]]
    acceptance = {0: [], 1: []}  # 1 for even and 0 for odd

    # PT main loop
    for pt_iter in trange(n_PT_steps):
        even_or_odd = pt_iter % 2 == 0

        path_x_prev, path_x_next, sigmas_prev, sigmas_next = _get_adjacents(
            path_x, sigmas, parity_pt_iter=even_or_odd
        )

        # fmt: off
        path_x_prev, path_x_next, accepted = _crepe_iter(
            path_x_prev, path_x_next,
            sigmas_prev, sigmas_next,
            model,
            n_steps_proposal=1,
            log_pot_sigma=log_pot_sigma,
            proposal_kernel=selected_proposal
        )
        # fmt: on

        # inplace update of path_x
        _update_adjacents(path_x, path_x_prev, path_x_next, parity_pt_iter=even_or_odd)

        if return_iterations:
            iterations.append(path_x[0].clone()[None])
        acceptance[int(even_or_odd)].append(accepted[None])

        # XXX uncomment to see the evolution of x_0
        if display_im and pt_iter % display_freq == 0:
            sample = path_x[0, 0]
            plot_field(sample)
            plt.show()

    if return_only_reconstructions:
        return path_x[0]

    return (
        torch.cat(iterations),
        path_x,
        {k: torch.cat(val, dim=0) for k, val in acceptance.items()},
    )


def _init_path(initial_noise, sigmas, model):
    sigmas_flipped = sigmas.flip(0)

    x_t = sigmas_flipped[0] * initial_noise
    path_x = [x_t[None]]

    for sigma_t, sigma_t_prev in zip(sigmas_flipped[:-1], sigmas_flipped[1:]):
        x_t = ddim_step(x_t, model.denoiser_fn, sigma_t, sigma_t_prev, eta=1.0)

        path_x.append(x_t[None])

    path_x = torch.cat(path_x)
    path_x = path_x.flip(0)

    return path_x


# fmt: off
def _crepe_iter(
    path_x_prev: Tensor, path_x_next: Tensor,
    sigmas_prev: Tensor, sigmas_next: Tensor,
    model: EDMWrapper,
    n_steps_proposal: int = 1,
    log_pot_sigma=None,
    proposal_kernel=None,
    inv_prob: InverseProblem = None,
):
    # TODO generalize of n_steps_proposal > 1

    # P(x, dy)
    path_y_prev, path_y_next, log_proposal_fwd, log_proposal_bwd = proposal_kernel(
        path_x_prev, path_x_next,
        sigmas_prev, sigmas_next,
        model,
        inv_prob=inv_prob,
    )
    accept_ratio = -(log_proposal_fwd + log_proposal_bwd)

    # P(y, dx)
    _, _, log_proposal_fwd, log_proposal_bwd = proposal_kernel(
        path_y_prev, path_y_next,
        sigmas_prev, sigmas_next,
        model,
        path_x_prev, path_x_next,
        inv_prob=inv_prob,
    )
    accept_ratio = accept_ratio + (log_proposal_fwd + log_proposal_bwd)

    # pi(y) / pi(x) for the interval [Y_prev, X_next]
    _, _, log_dm_fwd, log_dm_bwd = _dm_fwd_bwd(
        path_y_prev, path_x_next,
        sigmas_prev, sigmas_next,
        model,
        path_y_prev, path_x_next,
    )
    accept_ratio = accept_ratio + (log_dm_bwd - log_dm_fwd)

    # pi(x) / pi(y) for the interval [X_prev, Y_next]
    _, _, log_dm_fwd, log_dm_bwd = _dm_fwd_bwd(
        path_x_prev, path_y_next,
        sigmas_prev, sigmas_next,
        model,
        path_x_prev, path_y_next,
    )
    accept_ratio = accept_ratio - (log_dm_bwd - log_dm_fwd)

    # intermediate pots
    # TODO used cached values of denoisers
    accept_ratio = accept_ratio + (
          log_pot_sigma(path_y_prev, sigmas_prev)
        - log_pot_sigma(path_x_prev, sigmas_prev)
    )
    accept_ratio = accept_ratio + (
          log_pot_sigma(path_y_next, sigmas_next)
        - log_pot_sigma(path_x_next, sigmas_next)
    )
    # fmt: on

    log_u = torch.rand_like(accept_ratio).log()

    accepted = log_u < accept_ratio
    # NOTE broadcast ``accepted`` to shape (n_steps, batch, 1, 1, ...)
    # for compatibility of operations between it and path
    while accepted.ndim < path_x_prev.ndim:
        accepted = accepted.unsqueeze(-1)

    path_x_prev = torch.where(accepted, path_y_prev, path_x_prev)
    path_x_next = torch.where(accepted, path_y_next, path_x_next)

    return path_x_prev, path_x_next, accepted.squeeze()


def _dm_fwd_bwd(
    path_x_prev: Tensor,
    path_x_next: Tensor,
    sigmas_prev: Tensor,
    sigmas_next: Tensor,
    model: EDMWrapper,
    path_y_prev: Tensor = None,
    path_y_next: Tensor = None,
    **kwargs, # added for compatibility with other proposal kernels
):
    # NOTE log prob are of shape (n_steps, batch)
    x_ndim = len(path_x_next.shape[2:])

    # forward
    std_fwd = (sigmas_next**2 - sigmas_prev**2).sqrt()
    if path_y_next is None:
        path_y_next = path_x_prev + std_fwd * torch.randn_like(path_x_prev)

    log_proposal_fwd = Independent(
        Normal(loc=path_x_prev, scale=std_fwd), reinterpreted_batch_ndims=x_ndim
    ).log_prob(path_y_next)

    # backward
    denoised = _get_denoised(path_x_next, sigmas_next, model)
    mean_bwd, std_bwd = bridge_stats(
        path_x_next, denoised, sigmas_next, sigmas_prev, eta=1.0
    )
    if path_y_prev is None:
        path_y_prev = mean_bwd + std_bwd * torch.randn_like(path_x_next)

    log_proposal_bwd = Independent(
        Normal(loc=mean_bwd, scale=std_bwd), reinterpreted_batch_ndims=x_ndim
    ).log_prob(path_y_prev)

    return path_y_prev, path_y_next, log_proposal_fwd, log_proposal_bwd


AVAILABLE_PROPOSALS = {
    "prior": _dm_fwd_bwd,
}
