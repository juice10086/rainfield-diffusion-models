import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_PATH))


from pathlib import Path

import hydra
from omegaconf import DictConfig

import torch

from dm_ps.gaussian_process.utils import create_inv_prob
from dm_ps.gaussian_process.one_dim import GaussianProcess1D
from dm_ps.models.gaussian_priors import DenoiserGaussianEDM

from dm_ps.experiments_utils import fix_seed
from dm_ps.samplers import AVAILABLE_SAMPLERS


dtype = torch.float64
torch.set_default_dtype(dtype)

config_path = str(REPO_PATH / "configs")
config_name = "1d_gaussian_processes"


@hydra.main(config_path=config_path, config_name=config_name)
def run_sampler(cfg: DictConfig):
    device = cfg.device
    fix_seed(cfg.seed)
    torch.set_default_device(device)

    save_dir = Path(str(cfg.save_dir))
    save_dir.mkdir(exist_ok=True)

    # generate field
    # --- int experiment
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

    n_recs = cfg.n_reconstructions

    model = DenoiserGaussianEDM(dist.loc, dist.covariance_matrix, device=device)

    initial_noise = torch.randn((n_recs, n_samples))

    sampler_name = cfg.sampler.name
    sampler = AVAILABLE_SAMPLERS[sampler_name]

    reconstructions = sampler(
        initial_noise, inv_prob, model, **cfg.sampler.parameters_gp_1d
    )

    torch.save(reconstructions, save_dir / f"{sampler_name}_{n_recs}.pt")


if __name__ == "__main__":
    run_sampler()
