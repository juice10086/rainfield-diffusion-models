import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_PATH))


import time
from pathlib import Path
import hydra
from omegaconf import DictConfig

import torch
import numpy as np
from training.rain_dataset import RainDataset

from dm_ps.samplers import AVAILABLE_SAMPLERS
from dm_ps.models.edm_wrapper import DecoderFreeEDMOpenMRG
from dm_ps.inv_prob.metrics import diff_cum_rain_rate, mae, pearson_corr, rmse
from dm_ps.inv_prob.create_inv_prob_openmrg import (
    create_inv_prob,
    plot_field,
)

from dm_ps.experiments_utils import fix_seed


dtype = torch.float32

config_path = str(REPO_PATH / "configs")
config_name = "openmrg_real_cmls"


@hydra.main(config_path=config_path, config_name=config_name)
def run_sampler(cfg: DictConfig):
    device = cfg.device

    fix_seed(cfg.seed)
    torch.set_default_device(cfg.device)
    torch.set_default_dtype(dtype)
    torch.cuda.empty_cache()

    # set paths
    save_folder = Path(cfg.save_dir)

    # init model
    model: DecoderFreeEDMOpenMRG = DecoderFreeEDMOpenMRG(
        device=cfg.device,
        **cfg.model,
    )

    # load dataset
    dataset = RainDataset(dtype=dtype, **cfg.dataset)

    # load positions of links
    links = torch.from_numpy(np.load(cfg.links.path_links)).to(device=device)

    selected_field_idx = (
        cfg.field_idx if not cfg.run_on_all_fields else range(len(dataset.data))
    )

    for idx in selected_field_idx:
        field, *_ = dataset[[idx]]

        # create inverse problem
        inv_prob = create_inv_prob(
            field,
            obs_std=cfg.obs_std,
            device=cfg.device,
            links=links,
            dtype=torch.float32,
            non_negative_obs=cfg.non_negative_obs,
            path_cst_a_power_b=cfg.links.path_cst_a_power_b,
        )

        # solve problem
        initial_noise = torch.randn((cfg.n_samples, *model.x_shape))
        sampler = AVAILABLE_SAMPLERS[cfg.sampler.name]

        if cfg.reporting.print_runtime:
            start_time = time.perf_counter()

        samples = sampler(
            initial_noise, inv_prob, model, **cfg.sampler.parameters_real_cmls
        )

        if cfg.reporting.print_runtime:
            end_time = time.perf_counter()
            print(f"runtime (s)       : {end_time - start_time}")

        # save reconstructions
        torch.save(samples, save_folder / f"field_idx_{idx}.pt")

        if cfg.reporting.print_metrics:
            x_ref = inv_prob.x_ref[0]
            print("rmse               :", rmse(samples, x_ref).mean(dim=0))
            print("mae                :", mae(samples, x_ref).mean(dim=0))
            print("pearson_corr       :", pearson_corr(samples, x_ref).mean(dim=0))
            print(
                "diff_cum_rain_rate :",
                diff_cum_rain_rate(samples, x_ref).mean(dim=0),
            )
            print(
                "llh x_ref - llh rec:",
                (
                    inv_prob.log_pot_batch(samples)
                    - inv_prob.log_pot_batch(inv_prob.x_ref)
                ).mean(),
            )

        if cfg.reporting.plot_reconstruction:
            import matplotlib.pyplot as plt

            shift = cfg.reporting.plot_shift
            n_plotted_samples = min(cfg.reporting.n_plotted_samples, cfg.n_samples)
            fig, axes = plt.subplots(
                1, 2 + n_plotted_samples, sharex=True, sharey=True
            )

            axes[0].set_title("Ref")
            axes[1].set_title("Obs")

            im_plot = plot_field(inv_prob.x_ref, ax=axes[0])

            plot_field(inv_prob.x_ref, ax=axes[1])
            for link in links:
                link = link.cpu()
                axes[1].plot(
                    link[:, 0],
                    link[:, 1],
                    color="red",
                    marker="o",
                    markersize=3,
                    linewidth=1,
                )

            plotted_samples = samples[shift : n_plotted_samples + shift]
            for sample, ax, i in zip(
                plotted_samples, axes[2:], range(len(plotted_samples))
            ):
                plot_field(sample, ax=ax)
                ax.set_title(f"rec {i}")

            fig.colorbar(
                im_plot,
                ax=axes.ravel().tolist(),
                orientation="horizontal",
                label="Rain intensity in [mm/h]",
            )

            fig.savefig(
                save_folder / f"field_idx_{idx}{cfg.reporting.plot_suffix}",
                bbox_inches="tight",
            )
            plt.close(fig)

        print(f"Finished-field-{idx}")

    print(f"Finished {cfg.sampler.name}")


if __name__ == "__main__":
    run_sampler()
