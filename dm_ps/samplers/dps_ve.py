from tqdm import tqdm
import torch
from torch import Tensor
from dm_ps.bridge import bridge_stats
from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.inv_prob.base import InverseProblem


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
    guidance: str = 'likelihood',
    generator=None,
) -> Tensor:
    """DPS with gradients through the denoiser and the observation likelihood.

    ``guidance='residual'`` retains the original rain experiment's negative
    residual-norm heuristic. Radio experiments use ``'likelihood'``, including
    each measurement's noise variance, with a separately tuned gamma.
    """
    if n_steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError('Require n_steps>=2, 0<sigma_min<sigma_max and rho>0')
    if not 0 <= eta <= 1 or gamma < 0:
        raise ValueError('Require eta in [0,1] and gamma>=0')
    if guidance not in ('likelihood', 'residual'):
        raise ValueError('guidance must be likelihood or residual')

    timesteps = model.mk_sigmas_fn(
        n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )
    flipped_timesteps = timesteps.flip(0)

    x_t = (flipped_timesteps[0] * initial_noise).detach()
    for idx, (sigma_t, sigma_t_prev) in tqdm(
        enumerate(zip(flipped_timesteps[:-1], flipped_timesteps[1:]))
    ):
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            x_0 = model.denoiser_fn(x_t, sigma_t)
            if guidance == 'likelihood':
                loss = inverse_problem.log_pot(model.decode(x_0))
            else:
                loss = -torch.norm(inverse_problem.obs - inverse_problem.H_func.H(model.decode(x_0)))
            (grad,) = torch.autograd.grad(loss, x_t)
        if not torch.isfinite(grad).all():
            raise FloatingPointError('Non-finite likelihood gradient; check noise scales and gamma')

        with torch.no_grad():
            if normalize_grad:
                grad = grad / grad.square().sum(dim=(-3, -2, -1), keepdim=True).sqrt().clamp_min(1e-12)

            mean, std = bridge_stats(x_t, x_0, sigma_t, sigma_t_prev, eta=eta)
            noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
            x_t = mean + gamma * grad + std * noise
            if not torch.isfinite(x_t).all():
                raise FloatingPointError('Non-finite DPS state; decrease gamma or normalize the gradient')

        # XXX uncomment to see the evolution of x_0
        if display_im and idx % display_freq == 0:
            import matplotlib.pyplot as plt
            with torch.no_grad():
                sample = model.decode(model.denoiser_fn(x_t[[0]], sigma_t_prev))
            plt.imshow(sample[0, 0].detach().cpu().numpy())
            plt.show()

    with torch.no_grad():
        x_0 = model.denoiser_fn(x_t, sigma_t_prev)
        return model.decode(x_0)
