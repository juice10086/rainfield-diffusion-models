import torch

from dm_ps.inv_prob._siddon_algo import siddon_algo_2d
import matplotlib.pyplot as plt


plt.rcParams.update({"text.usetex": True, "font.family": "times"})
font_size = 12
fig, ax = plt.subplots(figsize=(8.27 / 2, 11.69 / 5))

# Grid size used in the Siddon operator illustration.
width, height = 6, 4

# Example CML segment crossing the grid.
start = torch.tensor([0.25, 1 + 2.25])
end = torch.tensor([4.1, 1.0])

line = torch.cat((start[None], end[None]))
points, dist, grid = siddon_algo_2d(start, end, width=width, height=height)

# Hide cells that are not crossed by the link.
grid[grid == 0.0] = torch.nan

ax.imshow(grid, cmap="Oranges", vmax=4.0 ** (1 / 2))

ax.plot(line[:, 0], line[:, 1], marker="o", color="blue", alpha=0.7)
ax.scatter(points[1:-1, 0], points[1:-1, 1], color="red", marker="o")

ax.set_xticks(torch.arange(width))
ax.set_yticks(torch.arange(height))


for val in torch.arange(width) - 1 / 2:
    ax.axvline(x=val, color="black", linewidth=1)
for val in torch.arange(height) - 1 / 2:
    ax.axhline(y=val, color="black", linewidth=1)

ax.set_xlabel("Width", fontsize=font_size)
ax.set_ylabel("Height", fontsize=font_size)

ax.xaxis.set_label_position("top")
ax.tick_params(
    axis="x",
    which="both",
    top=True,
    bottom=False,
    labeltop=True,
    labelbottom=False,
)

ax.set_aspect(1)
ax.grid(linestyle=":")

fig.savefig(
    f"illustration_cml_operator.pdf",
    bbox_inches="tight",
)
