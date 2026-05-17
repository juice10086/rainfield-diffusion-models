# How to run experiments

Run commands from the repository root. All experiment scripts use Hydra configs, so any config value can be overridden on the command line with `key=value`.

## 1D Gaussian process

Config: `configs/1d_gaussian_processes.yaml`

Generate reconstructions with `experiments/gp_one_dim_runner.py`:

```bash
python experiments/gp_one_dim_runner.py \
  sampler=daps \
  seed=12 \
  save_dir=experiments/out_gp_1d/seed_12
```

Run once per sampler you want to compare, for example `sampler=mgdm`, `sampler=reddiff`, `sampler=crepe`, `sampler=dps`, `sampler=tds`, or `sampler=mgps`.

Plot the comparison with `visualization/gp_one_dim.py`:

```bash
python visualization/gp_one_dim.py
```

The plotting script expects reconstruction files named `{sampler}_{n_reconstructions}.pt` in the configured `save_dir`.

## OpenMRG real CMLs

Script: `experiments/openmrg_real_cml_runner.py`

Config: `configs/openmrg_real_cmls.yaml`

Set the repository path expected by the script:

```bash
export REPO_PATH_JZ=$(pwd)
```

Run a real-CML experiment:

```bash
python experiments/openmrg_real_cml_runner.py \
  save_dir=experiments/out_openmrg_real \
  dataset.path=/path/to/test_rain_rate_f4.npy \
  model.network_pkl=/path/to/network-snapshot.pkl \
  links.path_links=/path/to/cml_coordinates_ls-all.npy \
  links.path_cst_a_power_b=/path/to/cst_a_power_b_openmrg.pt
```

Useful overrides:

```bash
python experiments/openmrg_real_cml_runner.py \
  field_idx='[587]' \
  n_samples=10 \
  reporting.print_runtime=true \
  reporting.plot_reconstruction=true
```

## OpenMRG simulated CMLs

Script: `experiments/openmrg_simulated_cml_runner.py`

Config: `configs/openmrg_simulated_cmls.yaml`

Run a simulated-CML experiment:

```bash
python experiments/openmrg_simulated_cml_runner.py \
  save_dir=experiments/out_openmrg_sim \
  dataset.path=/path/to/test_rain_rate_f4.npy \
  model.network_pkl=/path/to/network-snapshot.pkl
```

Use generated links from disk:

```bash
python experiments/openmrg_simulated_cml_runner.py \
  links.path_pre_generated_links=/path/to/simulated_seed_321_n_links_100_max_len_5.pt
```

Or generate links from config values:

```bash
python experiments/openmrg_simulated_cml_runner.py \
  links.n_links=100 \
  links.min_length=1 \
  links.max_length=5
```

## Common options

- Select fields: `field_idx='[587,3160]'`
- Run all test fields: `run_on_all_fields=true`
- Change sampler: `sampler=daps`, `sampler=mgdm`, etc.
- Print metrics: `reporting.print_metrics=true`
- Print runtime: `reporting.print_runtime=true`
- Save reconstruction plots: `reporting.plot_reconstruction=true`
- Change device: `device=cpu` or `device=cuda:0`
