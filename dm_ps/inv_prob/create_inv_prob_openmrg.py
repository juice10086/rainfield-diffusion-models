from pathlib import Path
from typing import Union

import torch
from torch import Tensor

from training.rain_dataset import RainDataset
from dm_ps.inv_prob._siddon_algo import siddon_algo_2d
from dm_ps.inv_prob.base import InverseProblem

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def create_inv_prob(
    field,
    obs_std,
    links: Tensor,
    device: str = "cpu",
    dtype=torch.float32,
    dataset: RainDataset = None,
    non_negative_obs: bool = False,
    # ---
    cst_a: torch.Tensor = None,
    power_b: torch.Tensor = None,
    # ---
    path_cst_a_power_b: str = None,
) -> InverseProblem:
    """"""

    # load fields use ``dataset`` if provided
    if isinstance(field, int) and dataset is None:
        field_rain_rate, *_ = RainDataset(
            path="data/test_rain_rate_f4.npy"
        )[[field]]
    elif isinstance(field, int):
        field_rain_rate, *_ = dataset[[field]]
    elif isinstance(field, Tensor):
        field_rain_rate = field
    else:
        raise ValueError("TODO")

    field_rain_rate = field_rain_rate.to(dtype=dtype, device=device)

    # load operator
    height, width = field_rain_rate.shape[-2], field_rain_rate.shape[-1]

    # load cst_a and power_b if available
    if path_cst_a_power_b is not None:
        cst_a, power_b = torch.load(
            path_cst_a_power_b, map_location=device, weights_only=False
        )
        cst_a = cst_a.to(dtype)
        power_b = power_b.to(dtype)

    H_func_cml = LineCensorOperator(
        device=device,
        dtype=dtype,
        links=links,
        height=height,
        width=width,
        # ---
        cst_a=cst_a,
        power_b=power_b,
    )

    # --- generate observation and log_pot function
    obs_cml = H_func_cml.H(field_rain_rate[None]).to(device)
    obs_cml = obs_cml + obs_std * torch.randn_like(obs_cml)

    if non_negative_obs:
        obs_cml = obs_cml.clamp(0.0, None)

    def log_pot_cml(x: Tensor):
        diff = obs_cml - H_func_cml.H(x)
        return -0.5 * torch.norm(diff) ** 2 / obs_std**2

    # NOTE needed for Replica exchange
    def log_pot_batch(x: Tensor):
        diff = obs_cml - H_func_cml.H(x)
        return -0.5 * ((diff / obs_std) ** 2).sum(dim=-1)

    return InverseProblem(
        obs=obs_cml,
        obs_img=None,
        obs_std=obs_std,
        log_pot=log_pot_cml,
        log_pot_batch=log_pot_batch,
        H_func=H_func_cml,
        x_ref=field_rain_rate,
    )


class LineCensorOperator:
    """Line Censor Operator (CML).

    Implements the power-law::

        obs = a * \int_{s in L} X(s)^b dL / |L|

    Parameters
    ----------
    links : Tensor of shape (n_links, 2, 2)
        Dimension 1 present start-end of the line
        whereas Dimension 2 are the xy-coordinate.

    cst_a: Tensor
        must have length equal to the number if links

    cst_b: Tensor
        must have length equal to the number if links
    """

    def __init__(
        self,
        links: torch.Tensor,
        device="cpu",
        dtype=torch.float32,
        height=48,
        width=36,
        # --- with shape (n_links,)
        cst_a: torch.Tensor = None,
        power_b: torch.Tensor = None,
    ):
        self.links = links
        self.device = device
        self.dtype = dtype
        self.height = height
        self.width = width

        n_links = links.shape[0]

        # params of the CML
        self.cst_a = cst_a
        if cst_a is None:
            self.cst_a = torch.ones((len(links),), dtype=dtype, device=device)
        self.power_b = power_b
        if power_b is None:
            self.power_b = torch.ones((len(links),), dtype=dtype, device=device)

        # measurement matrix
        measurement_matrix = torch.zeros(
            (n_links, height * width), dtype=dtype, device=device
        )

        for idx, link in enumerate(links):
            start, end = link

            _, dist, grid = siddon_algo_2d(start, end, height=height, width=width)

            measurement_matrix[idx] = grid.view(-1).to(device) / dist.sum()

        self.measurement_matrix = measurement_matrix.to(device)

    def H(self, x: Tensor):
        # XXX specific to openmrg where x (rain field) may have shape (..., c, h, w)
        *batch, c, h, w = x.shape
        batch_dim = [1 for _ in batch]

        # NOTE x must positive otherwise it cause nan when computing x**b
        # NOTE also small values causes instability in backward pass over the power
        #  added a small eps 1e-6 to avoid zeros
        x = x.clamp(0, None)
        x = x + 1e-6
        # ---

        # cast into right shape
        x_reshaped = x.view(*batch, 1, -1)
        cst_a = self.cst_a.view(*batch_dim, -1)
        power_b = self.power_b.view(*batch_dim, -1, 1)

        # vectorized implementation of: a * M @ x**b
        result = torch.einsum(
            "ij,...ij->...i", self.measurement_matrix, x_reshaped**power_b
        )
        result = cst_a * result

        return result

    def get_obs_as_img(self, obs):
        # XXX it is not obvious how to build obs as image here
        return


def plot_field(field: Tensor, ax=None, vlim=(1e-2, 250)):
    # field is in [-1, 1]

    if ax is None:
        ax = plt

    # for the colorbar
    if vlim is None:
        vlim = [0, 1]
    vmin, vmax = vlim

    field = field.cpu()
    field = field.squeeze()

    if field.ndim > 2:
        field = field.permute(1, 2, 0)

    # im_plot = ax.imshow(field, vmin=vmin, vmax=vmax)
    im_plot = ax.imshow(field, norm=LogNorm(vmin=vmin, vmax=vmax))
    if ax is None:
        ax.colorbar()

    return im_plot


def save_rain_field(field: Tensor, save_path: Union[str, Path]):
    import numpy as np
    from PIL import Image

    # from torchvision.transforms import v2 as tv_transforms

    field = (field + 1) / 2
    field = field.squeeze()

    if field.ndim > 2:
        field = field.permute(1, 2, 0)

    field = field.cpu().numpy()
    field = (field * 255).astype(np.uint8)

    img = Image.fromarray(field)
    img.save(save_path)

    # get_PIL = tv_transforms.ToPILImage()
    # get_PIL(field).save(save_path)


def compute_psnr(samples: torch.Tensor, references: torch.Tensor):
    # samples: B, H, W
    # references: 1, H, W or B, H, W
    B = samples.shape[0]

    if references.shape[0] == 1:
        references = references.repeat(B, 1, 1)

    diff = samples - references
    mse = torch.mean(diff**2, dim=(1, 2))

    peak = 1.0  # we normalize the image to (0., 1.)
    psnr = -10 * torch.log10(mse / peak)
    return psnr.cpu()


def generate_random_links(
    n_links: int,
    height: int = 48,
    width: int = 36,
    min_length: float = 5.0,
    max_length: float = 10.0,
    n_search_iter: int = 100,
    path_pre_generated_links=None,
    device=None,
):
    # NOTE if provided, `path_pre_generated_links` must be .pt file
    if path_pre_generated_links is not None:
        return torch.load(
            path_pre_generated_links, weights_only=False, map_location=device
        )

    all_links = []
    remaining_links = n_links
    height_width = torch.tensor([width - 1, height - 1], dtype=torch.float32)
    height_width = height_width[None, None, :]

    for _ in range(n_search_iter):
        # the shape meaning: (batch, point, xy-coordinates)
        links = height_width * torch.rand(size=(remaining_links, 2, 2))

        dist = links.diff(dim=1).norm(dim=(1, 2))
        query = (min_length <= dist) & (dist <= max_length)
        selected_links = links[query]

        all_links.append(selected_links)
        remaining_links = remaining_links - len(selected_links)

        if remaining_links == 0:
            break
    else:
        # TODO raise a warning in the case the loop completed without a beak
        pass

    return torch.cat(all_links)
