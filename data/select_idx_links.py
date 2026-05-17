from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def main():
    parser = ArgumentParser(
        description="Select non-empty test rain-field indices for experiments."
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=Path("data/test_rain_rate_f4.npy"),
        help="Path to the test rain-rate NumPy array.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp_settings"),
        help="Directory where selected indices are saved.",
    )
    parser.add_argument(
        "--output-name",
        default="selected_nnz_test_idx.npy",
        help="Output file name for selected indices.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of non-empty test fields to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for index selection.",
    )
    args = parser.parse_args()

    test_data = np.load(args.test_data_path)
    non_empty = ~(test_data == 0.0).all(axis=(-2, -1))
    (available_idx,) = np.where(non_empty)

    if args.n_samples > len(available_idx):
        parser.error(
            f"--n-samples={args.n_samples} exceeds the {len(available_idx)} "
            "available non-empty test fields."
        )

    rng = np.random.RandomState(args.seed)
    selected_idx = rng.choice(available_idx, size=args.n_samples, replace=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    np.save(output_path, selected_idx)
    print(f"Saved {len(selected_idx)} indices to {output_path}")


if __name__ == "__main__":
    main()
