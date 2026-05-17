import argparse
import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_PATH))


ALGO_DISPLAY_NAMES = {
    "mgdm": r"{\sc{MGDM}}",
    "daps": r"{\sc{DAPS}}",
    "reddiff": r"{\sc{RedDiff}}",
    "crepe": r"{\sc{CREPE}}",
    "dps": r"{\sc{DPS}}",
    "tds": r"{\sc{TDS}}",
    "mgps": r"{\sc{MGPS}}",
}


def render_table(results, output_format):
    import pandas as pd

    df = pd.DataFrame(results)

    if output_format == "csv":
        return df.to_csv(float_format="%.6f")

    if output_format == "markdown":
        return df.to_markdown(floatfmt=".2f")

    return df.to_latex(float_format="%.2f")


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for 1D GP reconstructions saved by gp_one_dim_runner.py."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_PATH / "configs",
        help="Directory containing the Hydra config files.",
    )
    parser.add_argument(
        "--config-name",
        default="1d_gaussian_processes",
        help="Hydra config name to compose.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override. May be passed multiple times, e.g. --override seed=35.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing {algo}_{n_reconstructions}.pt files. Defaults to cfg.save_dir.",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=list(ALGO_DISPLAY_NAMES),
        choices=list(ALGO_DISPLAY_NAMES),
        help="Algorithms to include in the metric table.",
    )
    parser.add_argument(
        "--q-left",
        type=float,
        default=0.05,
        help="Left posterior quantile used in the quantile-distance metric.",
    )
    parser.add_argument(
        "--q-right",
        type=float,
        default=0.95,
        help="Right posterior quantile used in the quantile-distance metric.",
    )
    parser.add_argument(
        "--n-slices",
        type=int,
        default=1000,
        help="Number of slices used for sliced Wasserstein distance.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of jobs used for sliced Wasserstein distance.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path where the rendered metric table is written.",
    )
    parser.add_argument(
        "--format",
        choices=("latex", "csv", "markdown"),
        default="latex",
        help="Output table format.",
    )
    args = parser.parse_args()

    if not 0.0 < args.q_left < args.q_right < 1.0:
        parser.error("Expected 0 < --q-left < --q-right < 1.")

    import torch
    import tqdm
    from hydra import compose, initialize_config_dir
    from torch.distributions import Normal

    from dm_ps.experiments_utils import fix_seed
    from dm_ps.gaussian_process.one_dim import (
        GaussianProcess1D,
        PosteriorGaussianProcess1DLineInt,
    )
    from dm_ps.gaussian_process.utils import (
        compute_sliced_wasserstein,
        create_inv_prob,
    )

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    with initialize_config_dir(
        version_base=None, config_dir=str(args.config_dir.resolve())
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)

    fix_seed(cfg.seed)
    torch.set_default_device(cfg.device)

    results_dir = args.results_dir or Path(str(cfg.save_dir))
    n_recs = cfg.n_reconstructions

    n_samples = cfg.prior.n_samples
    xlim = list(cfg.prior.xlim)
    t_range = torch.linspace(*xlim, n_samples)

    characteristic_length = cfg.prior.characteristic_length
    gp = GaussianProcess1D(characteristic_length)
    dist = gp.eval_gp(t_range)
    field = dist.sample((1,))

    intervals = torch.tensor(list(cfg.intervals))
    inv_prob = create_inv_prob(
        field,
        t_range,
        intervals,
        cfg.std_obs,
        device=cfg.device,
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

    marginals = Normal(post.mean, post.covariance_matrix.diag().sqrt())
    post_q_left = marginals.icdf(torch.tensor(args.q_left))
    post_q_right = marginals.icdf(torch.tensor(args.q_right))
    ref_samples = post.sample((n_recs,))

    d_results = {}
    for algo in tqdm.tqdm(args.algos):
        recon_path = results_dir / f"{algo}_{n_recs}.pt"
        recons = torch.load(recon_path, weights_only=False)

        mean = recons.mean(dim=0)
        approx_q_left = recons.quantile(args.q_left, dim=0)
        approx_q_right = recons.quantile(args.q_right, dim=0)

        dist_sw = compute_sliced_wasserstein(
            ref_samples, recons, n_jobs=args.n_jobs, n_slices=args.n_slices
        )
        dist_mean = torch.norm(mean - post.loc)
        dist_q_left = torch.norm(approx_q_left - post_q_left)
        dist_q_right = torch.norm(approx_q_right - post_q_right)

        d_results[ALGO_DISPLAY_NAMES[algo]] = {
            "sw": dist_sw.item(),
            "mean": dist_mean.item(),
            "q_left": dist_q_left.item(),
            "q_right": dist_q_right.item(),
        }

    table = render_table(d_results, args.format)
    print(table)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)


if __name__ == "__main__":
    main()
