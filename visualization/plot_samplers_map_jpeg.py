import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_PATH))

import torch
import numpy as np

import os
from pathlib import Path

from training.rain_dataset import RainDataset

from visualization._map_results import plot_rain_rec


results_dir = Path("/data/tii/rain/real_cml_openmrg")

# load test dataset
dataset = RainDataset(
    path="data/test_rain_rate_f4.npy"
)


# NOTE use this for loading real cmls
links = torch.from_numpy(
    np.load(
        "data/cml_coordinates_ls-all.npy"
    )
).to(device="cpu")
# ---

# NOTE and use this for loading the pre-generated links
# path_pre_generated_links = f"data/exp_settings/simulated_seed_321_n_links_25_max_len_20.pt"
# links = generate_random_links(
#     n_links=None,
#     device="cpu",
#     path_pre_generated_links=path_pre_generated_links,
# )
# ---

# NOTE use int(sys.argv[1]) to quickly run it as CLI
idx_selected_field = 3390  


save_path = Path(f"./viz")
save_path.mkdir(exist_ok=True)


resolution = "h"
name_algos = {
    "mgdm": r"{\sc{MGDM}}",
    "daps": r"{\sc{DAPS}}",
    "reddiff": r"{\sc{RedDiff}}",
    "dps": r"{\sc{DPS}}",
    "tds": r"{\sc{TDS}}",
    "mgps": r"{\sc{MGPS}}",
    "idw": r"{\sc{IDW}}",
    "gmz": r"{\sc{GMZ}}",
    "ok": r"{\sc{OK}}",
}


vlim = [1e-1, 1e2]

for sampler in name_algos:
    # select the last run
    run_name = sorted(os.listdir(results_dir / sampler))[-1]
    sampler_dir_results = results_dir / sampler / run_name

    f = f"field_idx_{idx_selected_field}.pt"

    samples = torch.load(sampler_dir_results / f, map_location="cpu")

    save_path_field = save_path / f"{idx_selected_field}"
    save_path_field.mkdir(exist_ok=True)

    plot_rain_rec(
        samples[0],
        links,
        resolution=resolution,
        title=f"{name_algos[sampler]}",
        vlim=vlim,
        path_save=save_path_field / f"{sampler}.jpeg",
    )


x_ref, *_ = dataset[[idx_selected_field]]
plot_rain_rec(
    x_ref,
    links,
    resolution=resolution,
    title=r"{Reference}",
    vlim=vlim,
    path_save=save_path_field / f"ref.jpeg",
)
