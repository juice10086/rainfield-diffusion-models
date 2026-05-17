import torch
import numpy as np

import matplotlib.pyplot as plt


device = "cuda:1"
seed = 11235

torch.manual_seed(seed)
rng = np.random.RandomState(seed)

n_selected_samples = 5_000

gen_data = torch.load(
    "put/the/path/to/pre-generated-rain-fields",
    map_location="cpu",
)

real_data = np.load("data/train_rain_rate_f4.npy")
real_data = real_data[rng.choice(len(real_data), n_selected_samples, replace=False)]
real_data = torch.from_numpy(real_data).to(torch.float32)


plt.rcParams.update({"text.usetex": True, "font.family": "times"})
font_size = 12

fig, axes = plt.subplots(1, 2, figsize=(8.27 / 2, 11.69 / 5))


arr_bins = np.linspace(0, max(real_data.max(), gen_data.max()), num=50)

axes[0].hist(
    real_data.reshape(-1),
    label="Reference",
    bins=arr_bins,
    alpha=1.0,
    density=True,
    histtype="step",
    color="blue",
)
axes[0].hist(
    gen_data.reshape(-1),
    label="Generated",
    bins=arr_bins,
    alpha=1.0,
    density=True,
    color="Red",
    edgecolor="Red",
)

axes[0].set_yscale("log")
axes[0].set_xlabel("Rain intensity (mm/h)", fontsize=font_size)
axes[0].set_yticks([1e-7, 1e-4, 1e-1])
axes[0].yaxis.set_tick_params(labelsize=10)
axes[0].xaxis.set_tick_params(labelsize=10)

cum_rain_gen = gen_data.sum(dim=(-2, -1))
cum_rain_real = real_data.sum(dim=(-2, -1))

arr_bins = np.linspace(0, max(cum_rain_real.max(), cum_rain_gen.max()), num=50)

axes[1].hist(
    cum_rain_real,
    label="Reference",
    bins=arr_bins,
    alpha=1.0,
    density=True,
    histtype="step",
    color="blue",
)
axes[1].hist(
    cum_rain_gen,
    label="Generated",
    bins=arr_bins,
    alpha=1.0,
    density=True,
    color="Red",
    edgecolor="Red",
)
axes[1].set_yscale("log", base=10)
axes[1].set_xlabel("Cumulative rain (mm/h)", fontsize=font_size)
axes[1].set_yticks([1e-5, 1e-4, 1e-3], minor=True)
axes[1].set_yticks([1e-5, 1e-4, 1e-3])
axes[1].yaxis.set_tick_params(labelsize=10)
axes[1].xaxis.set_tick_params(labelsize=10)


# Build one shared legend for both histograms.
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

_, indices = np.unique(labels, return_index=True)
lines = [lines[i] for i in indices]
labels = [labels[i] for i in indices]
indices = np.argsort(labels)
lines = [lines[i] for i in indices]
labels = [labels[i] for i in indices]

fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.52, 1.12),
    ncol=2,
    columnspacing=1.0,
    fontsize=font_size,
)
fig.tight_layout()

fig.savefig(
    f"stats_dm_prior.pdf",
    bbox_inches="tight",
)
