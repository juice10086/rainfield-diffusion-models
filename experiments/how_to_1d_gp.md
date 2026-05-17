# 1D GP experiments

Run commands from the repository root.

## Requirements

The metrics CLI uses pandas to render the result table. Install it in the experiment environment if it is not already available:

```bash
pip install pandas

# for display the table in markdown
pip install tabulate
```

## Generate reconstructions

`experiments/gp_one_dim_runner.py` writes one file per sampler.

Use a local output directory:

```bash
python experiments/gp_one_dim_runner.py sampler=daps save_dir=experiments/out_gp_1d/seed_12
python experiments/gp_one_dim_runner.py sampler=mgdm save_dir=experiments/out_gp_1d/seed_12
python experiments/gp_one_dim_runner.py sampler=reddiff save_dir=experiments/out_gp_1d/seed_12
python experiments/gp_one_dim_runner.py sampler=crepe save_dir=experiments/out_gp_1d/seed_12
python experiments/gp_one_dim_runner.py sampler=dps save_dir=experiments/out_gp_1d/seed_12
python experiments/gp_one_dim_runner.py sampler=tds save_dir=experiments/out_gp_1d/seed_12
python experiments/gp_one_dim_runner.py sampler=mgps save_dir=experiments/out_gp_1d/seed_12
```

Each output is named `{sampler}_{n_reconstructions}.pt`, for example `daps_1000.pt`.

## Compute metrics

Compute the default metric table from the config `save_dir`:

```bash
python experiments/compute_metrics_1d_gp.py
```

Compute metrics from a custom results directory:

```bash
python experiments/compute_metrics_1d_gp.py \
  --results-dir experiments/out_gp_1d/seed_12
```

Evaluate a different seed by passing Hydra overrides. Use the same seed and result directory that were used to generate the reconstructions:

```bash
python experiments/gp_one_dim_runner.py \
  sampler=daps \
  seed=35 \
  save_dir=experiments/out_gp_1d/seed_35
```

```bash
python experiments/compute_metrics_1d_gp.py \
  --override seed=35 \
  --results-dir experiments/out_gp_1d/seed_35
```

Restrict the table to a subset of algorithms:

```bash
python experiments/compute_metrics_1d_gp.py \
  --algos daps mgdm reddiff \
  --results-dir experiments/out_gp_1d/seed_12
```

Write the table to disk:

```bash
python experiments/compute_metrics_1d_gp.py \
  --results-dir experiments/out_gp_1d/seed_12 \
  --output experiments/out_gp_1d/seed_12/metrics.tex
```

Use another table format:

```bash
python experiments/compute_metrics_1d_gp.py \
  --results-dir experiments/out_gp_1d/seed_12 \
  --format markdown \
  --output experiments/out_gp_1d/seed_12/metrics.md
```

## Useful options

- `--config-dir`: Hydra config directory. Defaults to `configs/`.
- `--config-name`: Hydra config name. Defaults to `1d_gaussian_processes`.
- `--override`: Hydra override. Can be passed multiple times.
- `--results-dir`: directory containing `{algo}_{n_reconstructions}.pt` files.
- `--algos`: algorithms to include. Defaults to `mgdm daps reddiff crepe dps tds mgps`.
- `--q-left` and `--q-right`: posterior quantiles used for quantile-distance metrics.
- `--n-slices` and `--n-jobs`: sliced-Wasserstein computation settings.
- `--format`: `latex`, `csv`, or `markdown`.
- `--output`: optional path where the rendered table is saved.
