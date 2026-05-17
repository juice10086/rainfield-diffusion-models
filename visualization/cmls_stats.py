import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_PATH))


import torch
from pynncml.single_cml_methods.power_law.pl_module import a_b_parameters

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "times"})
font_size = 12

metadata = np.load(f"data/cml_metadata.npz")


cmls_a_and_b = torch.tensor(
    [
        [el.item() for el in a_b_parameters(f, pol)]
        for f, pol in zip(metadata["frequency"], metadata["polarization"])
    ]
)


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(8.27 / 2, 11.69 / 7))

axes[0].scatter(metadata["length"], cmls_a_and_b[:, 0], s=8)
axes[0].set_title(r"Constant $a$")

axes[1].scatter(metadata["length"], cmls_a_and_b[:, 1], s=8)
axes[1].set_title(r"Power $b$")

fig.tight_layout()
fig.supxlabel("Length of the CML", y=-0.05)

fig.savefig(
    f"cmls_a_b_length.jpeg",
    dpi=250,
    bbox_inches="tight",
)


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(8.27 / 2, 11.69 / 7))

axes[0].scatter(metadata["frequency"], cmls_a_and_b[:, 0], s=8)
axes[0].set_title(r"Constant $a$")

axes[1].scatter(metadata["frequency"], cmls_a_and_b[:, 1], s=8)
axes[1].set_title(r"Power $b$")

fig.tight_layout()
fig.supxlabel("Frequency of the CML", y=-0.05)

fig.savefig(
    f"cmls_a_b_frequency.jpeg",
    dpi=250,
    bbox_inches="tight",
)
