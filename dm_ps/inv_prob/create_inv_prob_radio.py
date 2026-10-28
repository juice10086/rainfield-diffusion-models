"""Target-frequency RSS/intensity point observations, never CML line integrals.

Coordinates use pixel units (x=column, y=row). Integer coordinates address pixel
centres. Fractional coordinates use bilinear interpolation of adjacent pixels.
Values and noise standard deviations must share the map's normalization.
"""

import torch
from torch import Tensor

from dm_ps.inv_prob.base import InverseProblem


class PointSamplingOperator:
    def __init__(self, positions, height, width, device='cpu', dtype=torch.float32):
        if height < 1 or width < 1:
            raise ValueError('Map dimensions must be positive')
        self.height, self.width = int(height), int(width)
        points = torch.as_tensor(positions, device=device, dtype=dtype).detach().clone()
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            raise ValueError('positions must have nonempty shape [N,2] in (x,y) pixel coordinates')
        if not torch.isfinite(points).all():
            raise ValueError('Receiver positions must be finite')
        x, y = points.unbind(-1)
        if ((x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)).any():
            raise ValueError('Receiver positions must lie inside the map')
        self.positions = points
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp(max=width - 1), (y0 + 1).clamp(max=height - 1)
        dx, dy = x - x0, y - y0
        self.indices = torch.stack([y0 * width + x0, y0 * width + x1,
                                    y1 * width + x0, y1 * width + x1], dim=-1)
        self.weights = torch.stack([(1 - dx) * (1 - dy), dx * (1 - dy),
                                    (1 - dx) * dy, dx * dy], dim=-1)

    @classmethod
    def from_mask(cls, mask, **kwargs):
        mask = torch.as_tensor(mask)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2 or not ((mask == 0) | (mask == 1)).all():
            raise ValueError('Observation mask must be binary [H,W] or [1,H,W]')
        rows, columns = torch.where(mask.bool())
        return cls(torch.stack([columns, rows], dim=-1), *mask.shape, **kwargs)

    def H(self, x: Tensor):
        """Map [...,1,H,W] to predicted measurements [...,N]."""
        if x.ndim < 3 or tuple(x.shape[-3:]) != (1, self.height, self.width):
            raise ValueError(f'Expected [...,1,{self.height},{self.width}], got {tuple(x.shape)}')
        if not x.is_floating_point():
            raise ValueError('Candidate maps must be floating point')
        flat = x.reshape(*x.shape[:-3], self.height * self.width)
        values = flat[..., self.indices.to(x.device)]
        return (values * self.weights.to(device=x.device, dtype=x.dtype)).sum(-1)

    def support_mask(self):
        """Pixels contributing nonzero weight to any observation, [1,H,W]."""
        mask = torch.zeros(self.height * self.width, dtype=torch.bool, device=self.positions.device)
        mask[self.indices[self.weights > 0]] = True
        return mask.reshape(1, self.height, self.width)


def create_inv_prob(field=None, obs_std=0.01, *, mask=None, positions=None,
                    observations=None, height=None, width=None, valid_mask=None,
                    device='cpu', dtype=torch.float32, generator=None):
    """Build a single scene's target-frequency Gaussian likelihood.

    Supply measured ``observations`` [N] for inference: no truth or synthetic
    noise is then used. With observations=None, explicitly simulate measurements
    from ``field`` [1,H,W] and add Gaussian noise once. Reuse the returned object
    for both baseline and source-conditioned paths.
    """
    if (mask is None) == (positions is None):
        raise ValueError('Provide exactly one of mask or positions')
    reference = None
    if field is not None:
        reference = torch.as_tensor(field, device=device, dtype=dtype).detach().clone()
        if reference.ndim != 3 or reference.shape[0] != 1 or not torch.isfinite(reference).all():
            raise ValueError('field must be a finite [1,H,W] map of one scene')
        if height is not None and height != reference.shape[-2]:
            raise ValueError('height differs from field')
        if width is not None and width != reference.shape[-1]:
            raise ValueError('width differs from field')
        height, width = reference.shape[-2:]
    if mask is not None:
        operator = PointSamplingOperator.from_mask(mask, device=device, dtype=dtype)
        if height is not None and height != operator.height or width is not None and width != operator.width:
            raise ValueError('Observation mask and map dimensions differ')
    else:
        if height is None or width is None:
            raise ValueError('Provide height and width when no reference field is available')
        operator = PointSamplingOperator(positions, height, width, device=device, dtype=dtype)
    if valid_mask is not None:
        valid = torch.as_tensor(valid_mask, device=device)
        if valid.shape != operator.support_mask().shape or not ((valid == 0) | (valid == 1)).all():
            raise ValueError('valid_mask must be binary [1,H,W]')
        if (operator.support_mask() & ~valid.bool()).any():
            raise ValueError('Observation interpolation includes excluded pixels')
    n_obs = len(operator.positions)
    std = torch.as_tensor(obs_std, device=device, dtype=dtype).detach().clone()
    if std.ndim > 1 or std.numel() not in (1, n_obs) or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('obs_std must be finite and positive: scalar or [N]')
    std = std.reshape(-1).expand(n_obs).clone()
    if observations is None:
        if reference is None:
            raise ValueError('Measured observations or a simulation reference field are required')
        obs = operator.H(reference)
        obs = obs + std * torch.randn(obs.shape, device=device, dtype=dtype, generator=generator)
    else:
        obs = torch.as_tensor(observations, device=device, dtype=dtype).detach().clone()
        if obs.shape != (n_obs,) or not torch.isfinite(obs).all():
            raise ValueError(f'observations must be finite and have shape [{n_obs}]')

    def log_pot_batch(x):
        prediction = operator.H(x)
        residual = (prediction - obs.to(prediction)) / std.to(prediction)
        return -0.5 * residual.square().sum(dim=-1)

    def log_pot(x):
        return log_pot_batch(x).sum()

    return InverseProblem(obs=obs, obs_img=None, obs_std=std, H_func=operator,
                          log_pot=log_pot, log_pot_batch=log_pot_batch, x_ref=reference)
