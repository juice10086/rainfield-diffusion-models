import torch
from torch import Tensor

from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.inv_prob.create_inv_prob_openmrg import InverseProblem

import numpy as np
from pykrige.ok import OrdinaryKriging


def ok_map(
    # ---
    initial_noise: torch.Tensor,
    inverse_problem: InverseProblem,
    model: EDMWrapper = None,
    # ---
    variogram_model: str = "exponential",
    pseudo_inv=True,
):
    np_dtype = np.float64

    dtype, device = initial_noise.dtype, initial_noise.device
    height, width = initial_noise.shape[-2:]

    links: Tensor = inverse_problem.H_func.links
    obs = inverse_problem.obs

    # cml constants
    cst_a = inverse_problem.H_func.cst_a
    power_b = inverse_problem.H_func.power_b

    # ---
    y_obs = obs
    cml_positions = links.mean(dim=1)
    x_grid = np.arange(width, dtype=np_dtype)
    y_grid = np.arange(height, dtype=np_dtype)

    # account for the power law
    y_obs = (y_obs / cst_a) ** (1 / power_b)

    # cast as numpy
    cml_positions = cml_positions.detach().cpu().numpy().astype(np_dtype)
    y_obs = y_obs.detach().cpu().numpy().astype(np_dtype)

    # ---
    xy = cml_positions
    x_obs_pos = xy[:, 0]
    y_obs_pos = xy[:, 1]

    z_obs = y_obs

    OK = OrdinaryKriging(
        x_obs_pos,
        y_obs_pos,
        z_obs,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
        pseudo_inv=pseudo_inv,
    )
    z_hat, z_var = OK.execute("grid", x_grid, y_grid)

    # re cast as torch tensors
    map_ok = torch.from_numpy(z_hat).to(dtype=dtype, device=device)

    # keep the same output shape as DMs (batch, *x_shape)
    return map_ok.unsqueeze(0)
