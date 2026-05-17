# Agent Plan: Refactor `data/` preprocessing scripts and documentation

## Goal

Refactor the Python scripts in `data/` so they are reusable, self-explanatory command-line tools, then update `data/README.md` to document the full data-preparation workflow.

The final result should let a user reproduce the generated data artifacts without editing Python files manually.

## Scope

Work only on the data-preparation code and documentation:

- `data/preprocess.py`
- `data/openmrg_cml.py`
- `data/generate_simulated_cml.py`
- `data/select_idx_links.py`
- `data/README.md`

Do not change training, experiment, model, sampler, or visualization code unless a minimal import-path fix is required to make the data scripts runnable from the repository root.

## Current Context

The existing scripts were written as notebook-style exploratory scripts. They include plotting, printing, hard-coded paths, commented save logic, and environment-specific path hacks.

The current `data/README.md` describes the intended outputs at a high level, but it does not provide complete reproducible CLI commands.

Expected generated artifacts include:

- `train_rain_rate_f4.npy`
- `test_rain_rate_f4.npy`
- `train_log_rain_rate_f4.npy`
- `test_log_rain_rate_f4.npy`
- `cst_a_power_b_openmrg.pt`
- `exp_settings/simulated_seed_321_n_links_100_max_len_5.pt`
- `exp_settings/simulated_seed_321_n_links_25_max_len_20.pt`
- `exp_settings/selected_nnz_test_idx.npy`

## Refactoring Requirements

### General requirements for every script

1. Convert each script into a normal Python CLI entry point:
   - Use `argparse`.
   - Add a `main()` function.
   - Guard execution with `if __name__ == "__main__": main()`.

2. Remove exploratory behavior from default execution:
   - Remove notebook cell markers.
   - Remove plotting from the main workflow.
   - Remove ad hoc diagnostic prints unless they are concise status messages useful for CLI users.
   - Remove hard-coded absolute paths such as `sys.path.append(...)`.

3. Keep only the logic needed to load inputs, transform data, and write outputs.

4. All scripts that save files must accept a configurable output directory:
   - Use an argument such as `--output-dir`.
   - Create the directory if it does not exist.
   - Preserve the current default output locations under `data/` so existing workflows still work.

5. All file inputs should be configurable:
   - Use arguments such as `--input`, `--rain-rate-path`, `--metadata-path`, `--coordinates-path`, or `--test-data-path`.
   - Keep current paths as defaults where reasonable.

6. Prefer a "one-function" (or bulk of a sequence of instructions) implementation rather than implementing small function; Avoid over-using try-except blocks
   The code should be readable without requiring the README to explain every implementation detail.

7. Use `pathlib.Path` for filesystem paths.

8. Keep outputs deterministic where randomness is used:
   - Expose `--seed`.
   - Use the current seed defaults unless there is a strong reason to change them.

## Per-Script Requirements

### `data/preprocess.py`

Purpose: load raw OpenMRG radar rain-rate data, crop it, optionally log-transform it, split into train/test, and save NumPy arrays.

Required CLI behavior:

- Accept `--input` with default `data/rain_rate.npy`.
- Accept `--output-dir` with default `data/`.
- Accept crop dimensions, defaulting to height `48` and width `36`.
- Accept a train/test split setting equivalent to the current behavior:
  - Current behavior uses the last one fifth of samples as test data.
  - Preserve that default.
- Save raw rain-rate outputs:
  - `train_rain_rate_f4.npy`
  - `test_rain_rate_f4.npy`
- Save log-transformed outputs when requested:
  - `train_log_rain_rate_f4.npy`
  - `test_log_rain_rate_f4.npy`
- Provide a clear option for whether to save raw, log-transformed, or both.
  The default should preserve the artifacts listed in `data/README.md` if feasible.

Implementation notes:

- The log transform currently intended by comments is `np.log1p(rain_rate) / np.log(10)`.
- Keep dtype conversion to `np.float32` before saving.
- Do not show plots or histograms during normal execution.

### `data/openmrg_cml.py`

Purpose: compute OpenMRG CML power-law constants from CML metadata.

Required CLI behavior:

- Accept `--metadata-path` with default `data/cml_metadata.npz`.
- Accept `--output-dir` with default `data/`.
- Save `cst_a_power_b_openmrg.pt` in the selected output directory.

Implementation notes:

- Preserve the current computation using `pynncml.single_cml_methods.power_law.pl_module.a_b_parameters`.
- The saved tensor should keep the existing orientation expected by downstream code: `cmls_a_and_b.T`.
- Remove plotting and metadata exploration.
- Remove the unused coordinate plotting workflow unless it is required for the final output.

### `data/generate_simulated_cml.py`

Purpose: generate simulated CML link geometries for experiment settings.

Required CLI behavior:

- Accept `--n-links`.
- Accept `--max-len`.
- Accept `--seed` with default `321`.
- Accept `--output-dir` with default `data/exp_settings/`.
- Accept `--device` with a reasonable default.
  Prefer CPU by default unless the existing generation function requires CUDA.
- Save using the existing naming pattern:
  - `simulated_seed_{seed}_n_links_{n_links}_max_len_{max_len}.pt`

Implementation notes:

- Keep the current geometry defaults:
  - `height=48`
  - `width=36`
  - `min_length=1`
  - `n_search_iter=1000`
- Expose these as CLI options if doing so is simple and does not make the interface confusing.
- Preserve deterministic generation through `torch.manual_seed(seed)`.

### `data/select_idx_links.py`

Purpose: select indices of non-empty test rain fields for experiments.

Required CLI behavior:

- Accept `--test-data-path` with default `data/test_rain_rate_f4.npy`.
- Accept `--output-dir` with default `data/exp_settings/`.
- Accept `--output-name` with default `selected_nnz_test_idx.npy`.
- Accept `--n-samples` with default `1000`.
- Accept `--seed` with default `1234`.
- Save the selected indices under the selected output directory.

Implementation notes:

- Preserve the current filtering rule:
  - Select only samples where the full rain field is not all zeros.
- Preserve sampling without replacement.
- Validate that enough non-empty samples are available before sampling.
- Remove plotting and dataset visualization from default execution.

## README Requirements

Update `data/README.md` so it is a practical reproduction guide.

Include:

1. A short overview of the generated artifacts and what each script produces.
2. Required raw inputs:
   - `data/rain_rate.npy`
   - `data/cml_metadata.npz`
   - `data/cml_coordinates_ls-all.npy`, if still relevant
3. A note that `cml_coordinates_ls-all.npy` and `cml_metadata.npz` come from the original OpenMRG data and are not generated by these scripts unless that changes.
4. CLI commands for the complete workflow, for example:
   - preprocessing rain rates
   - computing CML power-law constants
   - generating the two simulated CML settings
   - selecting non-empty test indices
5. Show how to use a custom output directory for each script.
6. Mention important defaults:
   - crop size `48 x 36`
   - train/test split using the last one fifth as test
   - simulation seed `321`
   - selected-index seed `1234`

The README should describe usage, not exploratory analysis.

## Validation Checklist

After refactoring, verify the following from the repository root:

1. Do not run the script as they requires additional dependencies; just that the syntax is correct

2. Scripts can write to a custom output folder without editing source code.

3. Output filenames match the names documented in `data/README.md`.

4. Existing downstream expectations are preserved:
   - Rain-rate arrays are saved as `float32`.
   - CML power-law constants are saved as a transposed tensor.
   - Simulated CML filenames include seed, number of links, and max length.
   - Selected test indices are saved under `exp_settings/` by default.

5. No script runs plotting code during normal CLI execution.

6. No script depends on a user-specific absolute path.

## Deliverables

The completed implementation should include:

- Refactored CLI versions of the four Python scripts.
- Updated `data/README.md`.
- A brief summary of changed behavior and validation commands run.

