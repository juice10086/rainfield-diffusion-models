import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


seed = 1234567

torch.manual_seed(seed)
rng = np.random.RandomState(seed)

n_selected_samples = 5000

gen_data = torch.load(
    "put/the/path/to/pre-generated-rain-fields",
    map_location="cpu",
)

real_data = np.load("data/train_rain_rate_f4.npy")
real_data = real_data[rng.choice(len(real_data), n_selected_samples, replace=False)]
real_data = torch.from_numpy(real_data).to(torch.float32)


# Suppress low-intensity numerical residuals in generated fields.
gen_data[gen_data < 2e-2] = 0.0


plt.rcParams.update({"text.usetex": True, "font.family": "times"})
font_size = 26


def _set_ax_layout(ax, rm_frame=False):
    ax.set_xticks([])
    ax.set_yticks([])
    rm_frame and ax.set_frame_on(False)


n_rows = 3
n_cols = 4
ratios_cols = [1 for _ in range(n_cols)]

idx_selected_uncurated = rng.choice(n_selected_samples, n_cols * n_rows, replace=False)

# Select rainy examples from the top cumulative-rain fields.
threshold = min(200, n_selected_samples)
idx_selected_rainy = rng.choice(threshold, n_cols * n_rows, replace=False)


# %%
title = "Reference"  #  "DM prior"
data = real_data  # real_data  #  gen_data

cum_data = data.sum(dim=(-2, -1))
arg_sorted_cum_data = cum_data.sort().indices
data_quantile = data[arg_sorted_cum_data[-threshold:]]


fig, axes = plt.subplots(
    n_rows,
    ncols=2 * n_cols + 1,
    width_ratios=[*ratios_cols, 1 / 5, *ratios_cols],
    sharex=True,
    sharey=True,
    figsize=(8.27, 11.69 / 2.3),
)
fig.subplots_adjust(wspace=0.05, hspace=-0.55)

axes_uncurated = axes[:, :n_cols]

axes_rainy = axes[:, n_cols + 1 :]

[_set_ax_layout(ax) for ax in axes_uncurated.flatten()]
[_set_ax_layout(ax) for ax in axes_rainy.flatten()]
[_set_ax_layout(ax, rm_frame=True) for ax in axes[:, n_cols].flatten()]

for im, ax in zip(data[idx_selected_uncurated], axes_uncurated.flatten()):
    ax.imshow(im, norm=LogNorm(vmin=1e-2, vmax=100), cmap="Blues")

for im, ax in zip(data_quantile[idx_selected_rainy], axes_rainy.flatten()):
    ax.imshow(im, norm=LogNorm(vmin=1e-2, vmax=100), cmap="Blues")


fig.suptitle(title, y=0.83, x=0.51, fontsize=font_size)

fig.subplots_adjust(bottom=0.0)

fig.savefig(
    f"{'-'.join(title.lower().split(' '))}-samples.pdf",
    bbox_inches="tight",
)
