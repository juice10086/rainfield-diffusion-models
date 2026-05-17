# NOTE
#   lon (x-axis)
#   lat (y-axis)
# %%
import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_PATH))


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap

# ---
plt.rcParams.update({"text.usetex": True, "font.family": "times"})
font_size = 12
# ---

H, W = 48, 36

# Gothenburg (lon, lat) frame
TOP_LEFT_COORD = (11.39528, 58.04849)  # Approx North-West Europe
LOWER_RIGHT_COORD = (12.67385, 57.21925)  # Approx Middle East/North Africa


def plot_rain_rec(
    field, links, resolution="i", vlim=[1e-2, 240], path_save=None, title=None
):
    """
    resolution: Resolution of boundaries ('c', 'l', 'i', 'h', 'f')
    """

    field = field.squeeze()

    # --- internal utils
    def grid_coord_to_map(x, y):
        diff_lon = LOWER_RIGHT_COORD[0] - TOP_LEFT_COORD[0]
        diff_lat = TOP_LEFT_COORD[1] - LOWER_RIGHT_COORD[1]

        lon = (x / W) * diff_lon + TOP_LEFT_COORD[0]
        lat = (y / H) * diff_lat + LOWER_RIGHT_COORD[1]

        return lon, lat

    def draw_link(start, end):
        start_map = grid_coord_to_map(*start)
        end_map = grid_coord_to_map(*end)

        x_start, y_start = m(*start_map)
        x_end, y_end = m(*end_map)
        m.plot(
            [x_start, x_end],
            [y_start, y_end],
            # color="Orange",
            color="Red",
            linewidth=0.5,
            alpha=1.0,
            marker=".",
            markersize=1,
        )

    # ---

    # XXX
    lon_ul, lat_ul = TOP_LEFT_COORD
    lon_lr, lat_lr = LOWER_RIGHT_COORD

    # Initialize the figure
    # plt.figure(figsize=(8.27, 11.69 / 5))
    plt.figure(figsize=((W / H) * (11.69 / 5), 11.69 / 5))

    # plt.gca().set_aspect()

    # Create the Basemap instance
    m = Basemap(
        llcrnrlat=lat_lr,
        urcrnrlat=lat_ul,
        llcrnrlon=lon_ul,
        urcrnrlon=lon_lr,
        projection="merc",
        resolution=resolution,
    )

    img = m.imshow(
        field,
        origin="lower",
        cmap="Blues",
        alpha=1.0,
        norm=LogNorm(*vlim),
        interpolation="bilinear",
    )

    # Add visual elements
    m.drawcoastlines(linewidth=0.2, zorder=999)
    m.fillcontinents(color="gray", lake_color="black", alpha=0.1)

    # draw links
    for link in links:
        start, end = link
        draw_link(start, end)

    # ---
    if title:
        plt.title(title)

    if path_save:
        plt.margins(0, 0)
        plt.savefig(path_save, dpi=350, bbox_inches="tight")
        # plt.savefig(path_save, bbox_inches="tight", pad_inches=0.05)
        plt.close()


# for testing purposes
if __name__ == "__main__":
    from training.rain_dataset import RainDataset
    from dm_ps.inv_prob.create_inv_prob_openmrg import generate_random_links

    dataset = RainDataset(
        path="data/test_rain_rate_f4.npy"
    )
    cum_rain = dataset.data.sum(dim=(-3, -2, -1))
    # select 10 field
    ordered_index = cum_rain.sort(descending=True).indices[50:60]

    field_idx = 3383
    links = generate_random_links(
        n_links=15, height=48, width=36, min_length=1, max_length=5, device="cpu"
    )
    # ---

    for field_idx in ordered_index:
        x_ref, *_ = dataset[field_idx]

        plot_rain_rec(
            x_ref,
            links=links,
            resolution="i",
            path_save=f"./{field_idx}.jpeg",
        )


# %%
