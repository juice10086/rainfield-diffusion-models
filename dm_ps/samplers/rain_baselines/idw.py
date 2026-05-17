import torch
from torch import Tensor
from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.inv_prob.create_inv_prob_openmrg import InverseProblem


def idw_map(
    # ---
    initial_noise: torch.Tensor,
    inverse_problem: InverseProblem,
    model: EDMWrapper = None,
    # ---
    p=2,
    roi: float = None,
    eps: float = 1e-6,
    # --- when used with GMZ
    values=None,
    measurement_locations=None,
):
    device = initial_noise.device
    height, width = initial_noise.shape[-2:]

    links: Tensor = inverse_problem.H_func.links
    obs = inverse_problem.obs

    # cml constants
    cst_a = inverse_problem.H_func.cst_a
    power_b = inverse_problem.H_func.power_b

    # build inference grid
    x_grid = torch.arange(width, device=device)
    y_grid = torch.arange(height, device=device)
    y_mesh, x_mesh = torch.meshgrid(y_grid, x_grid, indexing="ij")
    grid = torch.stack([x_mesh, y_mesh], dim=0)

    # case where IDW is standalone
    if values is None and measurement_locations is None:
        values = obs
        measurement_locations = links.mean(dim=1)

    loc = measurement_locations
    # account for the power law
    values = (values / cst_a) ** (1 / power_b)

    rain_map = _base_idw(values, loc, grid, roi=roi, p=p, eps=eps)

    # keep the same output shape as DMs (batch, *x_shape)
    return rain_map.unsqueeze(0)


def _base_idw(
    values: Tensor,
    loc: Tensor,
    grid: Tensor,
    # ---
    roi=None,
    p=2,
    eps=1e-6,
):
    values = values.view(-1, 1, 1)
    loc = loc[:, :, None, None]
    grid = grid[None, :, :, :]

    dist = torch.sqrt(((loc - grid) ** 2).sum(dim=1))

    weights = 1.0 / (dist**p + eps)

    if roi is not None:
        weights[dist > roi] = 0.0

    weighted_sum = (weights * values).sum(dim=0)
    weight_sum = weights.sum(dim=0) + eps

    rain_map = weighted_sum / weight_sum

    return rain_map
