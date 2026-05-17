import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

from dm_ps.create_inv_prob import InverseProblem


class IntervalOperator:
    """"""

    def __init__(
        self,
        intervals: torch.Tensor,
        t_range: Tensor,
        device="cpu",
        dtype=torch.float32,
    ):
        # of shape (n_intervals, 2)
        self.intervals = intervals
        self.t_range = t_range

        self.device = device
        self.dtype = dtype

        n_intervals = intervals.shape[0]
        n_points = t_range.shape[0]
        measurement_matrix = torch.zeros(
            (n_intervals, n_points), dtype=dtype, device=device
        )

        for idx, interval in enumerate(intervals):
            start, end = interval

            *_, operator_i = _siddon_algo_1d(start, end, t_range)
            measurement_matrix[idx] = operator_i

        self.measurement_matrix = measurement_matrix.to(device)

    def H(self, x: Tensor):
        result = torch.einsum("ij,bj->bi", self.measurement_matrix, x)
        return result


def _siddon_algo_1d(start: Tensor, end: Tensor, t_range: Tensor):
    # NOTE t_range is assumed to be regularly spaced
    # TODO check for that explicitly
    step_size = t_range.diff()[0]
    t_range_start = t_range[0]

    # NOTE to catch silent bug when start and end fall outside t_range
    assert (t_range.min() <= start) and (
        end <= t_range.max()
    ), "Interval must be inside the grid"

    # map to integer grid
    left = (start - t_range_start) / step_size
    right = (end - t_range_start) / step_size

    intersection_points = torch.arange(left.ceil(), right.ceil())
    intersection_points = torch.cat(
        (left[None], intersection_points, right[None])
    ).unique()

    indices = intersection_points[:-1].floor().to(torch.int32)
    # remap to grid
    intersection_points = intersection_points * step_size + t_range_start
    dist = intersection_points.diff()

    operator = torch.zeros_like(t_range)
    operator[indices] = dist

    return intersection_points, dist, operator


def create_inv_prob(
    field: Tensor,
    t_range,
    intervals,
    obs_std,
    device,
    dtype,
    use_integral_for_obs: bool = False,
) -> InverseProblem:
    """"""
    # when ``use_integral_for_obs=True``, computes integral with ``scipy.interpolate.CubicSpline``, where the function is defined with ``field``

    H_func = IntervalOperator(
        intervals=intervals, t_range=t_range, device=device, dtype=dtype
    )

    # --- generate observation
    if not use_integral_for_obs:
        obs = H_func.H(field)
        obs = obs.squeeze(0)
    else:
        from scipy.interpolate import CubicSpline

        func = CubicSpline(t_range, field.squeeze())
        # NOTE integrate is not vectorized func, hence computing integrals via loop
        obs = torch.tensor(
            [func.integrate(start, end).item() for start, end in intervals],
            dtype=dtype,
            device=device,
        )
    obs = obs + obs_std * torch.randn_like(obs)
    # ---

    def log_pot(x: Tensor):
        diff = obs - H_func.H(x)
        return -0.5 * torch.norm(diff) ** 2 / obs_std**2

    def log_pot_batch(input):
        # input has shape (..., dim)
        Ax = torch.einsum("ij,...j->...i", H_func.measurement_matrix, input)
        return -0.5 * (((obs - Ax) / obs_std) ** 2).sum(dim=-1)

    return InverseProblem(
        x_ref=field,
        obs=obs,
        obs_std=obs_std,
        H_func=H_func,
        log_pot=log_pot,
        obs_img=None,
        log_pot_batch=log_pot_batch,
    )


def compute_W2_multivariate_normal(
    dist_1: MultivariateNormal, dist_2: MultivariateNormal
):
    mean_term = torch.norm(dist_1.mean - dist_2.mean) ** 2

    cov_1, cov_2 = dist_1.covariance_matrix, dist_2.covariance_matrix
    sqrt_cov_1 = torch_sqrtm(cov_1)

    trace_term = torch.trace(
        cov_1 + cov_2 - 2 * torch_sqrtm(sqrt_cov_1 @ cov_2 @ sqrt_cov_1)
    )

    return (mean_term + trace_term).sqrt()


def compute_sliced_wasserstein(
    dist_1: torch.Tensor, dist_2: torch.Tensor, n_slices: int = 100, n_jobs: int = 1
) -> float:
    """Compute sliced Wasserstein distance between two distributions.

    Assumes that both ``dist_1`` and ``dist_2`` have the same dimension.

    Parameters
    ----------
    dist_1 : Tensor

    dist_2 : Tensor

    n_slices : int, default=100
        The number of the considered random projections.

    Return
    ------
    sw_distance : float
    """
    import numpy as np
    from scipy.stats import wasserstein_distance
    from joblib import Parallel, delayed

    if dist_1.ndim > 2:
        dist_1 = dist_1.reshape(dist_1.shape[0], -1)
        dist_2 = dist_2.reshape(dist_2.shape[0], -1)

    projections = torch.randn(size=(n_slices, dist_1.shape[1]), device=dist_1.device)
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T

    dist_1_projected = dist_1_projected.cpu().numpy()
    dist_2_projected = dist_2_projected.cpu().numpy()

    job = delayed(lambda u, v: wasserstein_distance(u, v))
    return np.mean(
        Parallel(n_jobs)(
            job(d1, d2) for d1, d2 in zip(dist_1_projected, dist_2_projected)
        )
    )


def torch_sqrtm(M):
    """Compute the square root of symmetric matrix."""
    # NOTE: here, we use svd as it has proven to be more numerically stable
    # using eigh sometimes returns negative eigen value

    # XXX: implementation with eigh
    # eig_vals, P = torch.linalg.eigh(M)
    # return P @ (eig_vals.sqrt()[:, None] * P.T)

    U, S, Vh = torch.linalg.svd(M)
    return U @ (S[:, None].sqrt() * Vh)


def _test_siddon_1d():
    # NOTE for checking siddon algo
    from matplotlib.ticker import FormatStrFormatter

    n_samples = 10
    t_range = torch.linspace(-2, 2, n_samples)
    s, e = torch.tensor([-0.5, 1.7])

    intersection_points, dist, operator = _siddon_algo_1d(s, e, t_range)

    fig, ax = plt.subplots()
    ax.plot(t_range, torch.zeros_like(t_range))
    ax.scatter(intersection_points, torch.zeros_like(intersection_points), color="red")
    ax.grid()

    ax.set_xticks(t_range)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
