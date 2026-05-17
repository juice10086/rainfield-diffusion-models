import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_PATH))

from hydra import initialize_config_dir, compose

import torch
from torch import Tensor
from torch.distributions import Normal

from scipy.interpolate import CubicSpline

from dm_ps.gaussian_process.utils import create_inv_prob
from dm_ps.gaussian_process.one_dim import (
    GaussianProcess1D,
    PosteriorGaussianProcess1DLineInt,
)
from dm_ps.experiments_utils import fix_seed
import matplotlib.pyplot as plt


dtype = torch.float64
torch.set_default_dtype(dtype)


config_path = str(REPO_PATH / "configs")
config_name = "1d_gaussian_processes"


with initialize_config_dir(version_base=None, config_dir=config_path):
    cfg = compose(config_name=config_name, overrides=["seed=12"])


seed = cfg.seed
device = cfg.device
fix_seed(seed)
torch.set_default_device(device)

save_dir = Path(str(cfg.save_dir))

# Recreate the same 1D GP inverse problem used to generate the reconstructions.
n_samples = cfg.prior.n_samples
xlim = list(cfg.prior.xlim)
t_range = torch.linspace(*xlim, n_samples)

characteristic_length = cfg.prior.characteristic_length

gp = GaussianProcess1D(characteristic_length)
dist = gp.eval_gp(t_range)
field = dist.sample((1,))

std_obs = cfg.std_obs
li_intervals = list(cfg.intervals)
intervals = torch.tensor(li_intervals)

inv_prob = create_inv_prob(
    field,
    t_range,
    intervals,
    std_obs,
    device=device,
    dtype=dtype,
    use_integral_for_obs=True,
)

gp_post = PosteriorGaussianProcess1DLineInt(
    characteristic_length,
    intervals=intervals,
    obs=inv_prob.obs,
    std_obs=inv_prob.obs_std,
)
post = gp_post.eval_gp(t_range)
post_mean = post.loc


def _plot_inverse_problem(ax):
    for s, e in intervals:
        ax.axvspan(
            s, e, alpha=0.15, color="gray", linewidth=0, label="Observed interval"
        )
    for obs_val, (s, e) in zip(inv_prob.obs, intervals):
        ax.hlines(
            y=obs_val, xmin=s, xmax=e, linewidth=1, color="gray", linestyles="dashed"
        )


name_algos = {
    "crepe": r"{\sc{CREPE}}",
    "daps": r"{\sc{DAPS}}",
    "dps": r"{\sc{DPS}}",
    "mgdm": r"{\sc{MGDM}}",
    "mgps": r"{\sc{MGPS}}",
    "reddiff": r"{\sc{RedDiff}}",
    "tds": r"{\sc{TDS}}",
}
q_left, q_right = 0.05, 0.95

color = "Blue"

plt.rcParams.update({"text.usetex": True, "font.family": "times"})
font_size = 12
fig, axes = plt.subplots(
    2, (len(name_algos) + 1) // 2, figsize=(8.27, 11.69 / 4), sharey=True, sharex=True
)

ax = axes[0, 0]

t_refined_range = torch.linspace(t_range.min(), t_range.max(), 200)
func = CubicSpline(t_range, field[0])
field_refined = func(t_refined_range)

ax.plot(t_refined_range, field_refined, color="gray", label="Ground truth")

_plot_inverse_problem(ax)

ax.plot(t_range, post_mean, label="Mean", color=color)

marginals = Normal(post.mean, post.covariance_matrix.diag().sqrt())
ax.fill_between(
    t_range,
    marginals.icdf(torch.tensor(q_left)),
    marginals.icdf(torch.tensor(q_right)),
    color=color,
    alpha=0.1,
    linewidth=0,
    label=rf"${100 * q_left:.0f}\%-{100 * q_right:.0f}\%$ Quantile",
)

ax.set_title("True posterior")


# plot reconstructions
path_results = save_dir

for ax, algo in zip(axes.flatten()[1:], name_algos):
    recons: Tensor = torch.load(path_results / f"{algo}_1000.pt", weights_only=False)

    # plot mean
    ax.plot(t_range, recons.mean(dim=0), color=color)

    # plot quantiles
    ax.fill_between(
        t_range,
        recons.quantile(q_left, dim=0),
        recons.quantile(q_right, dim=0),
        color=color,
        alpha=0.1,
        linewidth=0,
    )

    _plot_inverse_problem(ax)

    ax.plot(t_refined_range, field_refined, color="gray")

    ax.set_title(f"{name_algos[algo]}")


# make common legend
import numpy as np

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

_, indices = np.unique(labels, return_index=True)
lines = [lines[i] for i in indices]
labels = [labels[i] for i in indices]
indices = np.argsort(labels)
lines = [lines[i] for i in indices]
labels = [labels[i] for i in indices]

fig.tight_layout()
fig.legend(
    lines,
    labels,
    ncols=4,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),
)

# save
fig.savefig(
    f"./gp-comparison-{seed}.jpeg",
    format="jpeg",
    dpi=300,
    bbox_inches="tight",
)
