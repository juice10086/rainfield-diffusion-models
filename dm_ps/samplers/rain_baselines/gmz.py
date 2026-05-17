# Code adapted from : https://github.com/haihabi/PyNNcml/blob/main/pynncml/multiple_cmls_methods/rain_field_reconstruction/gmz.py
import torch
from torch import Tensor

from dm_ps.samplers.rain_baselines.idw import _base_idw
from dm_ps.models.edm_wrapper import EDMWrapper
from dm_ps.inv_prob.create_inv_prob_openmrg import InverseProblem


def gmz_map(
    # ---
    initial_noise: torch.Tensor,
    inverse_problem: InverseProblem,
    model: EDMWrapper = None,
    # ---
    point_per_link=3,
    n_iter=10,
    p=2,
    roi=None,
    eps=1e-6,
    # ---
    return_loss_history: bool = False,
):
    device = initial_noise.device
    obs = inverse_problem.obs
    links: Tensor = inverse_problem.H_func.links

    height, width = initial_noise.shape[-2:]
    cst_a = inverse_problem.H_func.cst_a
    power_b = inverse_problem.H_func.power_b
    power_b_repeated = power_b.repeat_interleave(point_per_link)

    grid_x = torch.arange(width, device=device)
    grid_y = torch.arange(height, device=device)
    y_mesh, x_mesh = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack([x_mesh, y_mesh], dim=0)

    rain_links = obs.view(-1)
    points = _sample_points_on_links(links, point_per_link)
    points_flat = points.reshape(-1, 2)

    Z_links = rain_links / cst_a  #  ** (1 / power_b)
    Z_points = Z_links[:, None].expand(-1, point_per_link).reshape(-1)

    loss_history = []

    for _ in range(n_iter):

        rain_obs = Z_points ** (1 / power_b_repeated)
        Z_map = _base_idw(rain_obs, points_flat, grid, roi=roi, p=p, eps=eps)
        Z_map = Z_map.unsqueeze(0)

        Z_hat = _bilinear_sample(Z_map, points_flat, grid_x, grid_y)
        Z_hat = Z_hat.squeeze()
        Z_hat = Z_hat**power_b_repeated
        Z_hat = Z_hat.view(-1, point_per_link)

        if return_loss_history:
            loss = ((Z_points.view(-1, point_per_link) - Z_hat) ** 2).mean()
            loss_history.append(loss.detach())

        U = Z_links[:, None] - Z_hat.mean(dim=1, keepdim=True) + Z_hat
        U = torch.clamp(U, min=0.0)

        Z_points = U.reshape(-1)

    rain_map = Z_map

    # keep the same output shape as DMs (batch, *x_shape)
    rain_map = rain_map.unsqueeze(0)

    if return_loss_history:
        return rain_map, torch.stack(loss_history)

    return rain_map


def _sample_points_on_links(links, point_per_link):
    t = torch.linspace(0, 1, point_per_link, device=links.device)
    start = links[:, 0]
    end = links[:, 1]
    return (
        start[:, None, :] * (1 - t)[None, :, None] + end[:, None, :] * t[None, :, None]
    )


# taken from
# https://github.com/haihabi/PyNNcml/blob/84d878b0fb0b586a0f01c4d56721bbf926907c9b/pynncml/multiple_cmls_methods/rain_field_reconstruction/gmz.py#L111-L148
def _bilinear_sample(in_rain_map, points_set, grid_x, grid_y):
    point_set = points_set
    point_set_x = point_set[:, 0]
    point_set_y = point_set[:, 1]
    x_grid_vector = grid_x
    y_grid_vector = grid_y

    delta_x = point_set_x.unsqueeze(dim=-1) - x_grid_vector.unsqueeze(dim=0)
    delta_y = point_set_y.unsqueeze(dim=-1) - y_grid_vector.unsqueeze(dim=0)
    i = torch.argmin(torch.abs(delta_x), dim=1)
    is_pos = (torch.gather(delta_x, dim=1, index=i.reshape(-1, 1)) > 0).long().flatten()
    i_floor = i * is_pos + (i - 1) * (1 - is_pos)
    i_ceiling = i * (1 - is_pos) + (i + 1) * is_pos

    j = torch.argmin(torch.abs(delta_y), dim=1)
    js_pos = (torch.gather(delta_y, dim=1, index=j.reshape(-1, 1)) > 0).long().flatten()
    j_floor = j * js_pos + (j - 1) * (1 - js_pos)
    j_ceiling = j * (1 - js_pos) + (j + 1) * js_pos

    ff = in_rain_map[:, j_floor, i_floor]
    cf = in_rain_map[:, j_floor, i_ceiling]
    fc = in_rain_map[:, j_ceiling, i_floor]
    cc = in_rain_map[:, j_ceiling, i_ceiling]

    w_x_floor = (point_set_x - x_grid_vector[i_floor]) / (
        x_grid_vector[i_ceiling] - x_grid_vector[i_floor]
    )
    w_x_ceil = (x_grid_vector[i_ceiling] - point_set_x) / (
        x_grid_vector[i_ceiling] - x_grid_vector[i_floor]
    )
    w_y_floor = (point_set_y - y_grid_vector[j_floor]) / (
        y_grid_vector[j_ceiling] - y_grid_vector[j_floor]
    )
    w_y_ceil = (y_grid_vector[j_ceiling] - point_set_y) / (
        y_grid_vector[j_ceiling] - y_grid_vector[j_floor]
    )

    rain_point = (
        w_x_floor * w_y_floor * ff
        + w_x_ceil * w_y_floor * cf
        + w_x_floor * w_y_ceil * fc
        + w_x_ceil * w_y_ceil * cc
    )

    return rain_point.T
