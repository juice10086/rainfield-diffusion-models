from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def main():
    parser = ArgumentParser(
        description="Crop OpenMRG rain rates, split train/test data, and save arrays."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rain_rate.npy"),
        help="Path to the raw rain-rate NumPy array.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where processed arrays are saved.",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=48,
        help="Number of rows to keep from each rain field.",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=36,
        help="Number of columns to keep from each rain field.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of the final time steps to reserve for the test split.",
    )
    parser.add_argument(
        "--save",
        choices=("raw", "log", "both"),
        default="both",
        help="Choose whether to save raw rain rates, log-transformed rain rates, or both.",
    )
    args = parser.parse_args()

    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rain_rate = np.load(args.input)
    rain_rate = rain_rate[..., : args.crop_height, : args.crop_width]

    n_test = int(len(rain_rate) * args.test_fraction)
    if n_test <= 0 or n_test >= len(rain_rate):
        parser.error("--test-fraction leaves an empty train or test split.")

    train = rain_rate[:-n_test]
    test = rain_rate[-n_test:]

    if args.save in ("raw", "both"):
        train_path = output_dir / "train_rain_rate_f4.npy"
        test_path = output_dir / "test_rain_rate_f4.npy"
        np.save(train_path, train.astype(np.float32))
        np.save(test_path, test.astype(np.float32))
        print(f"Saved {train_path}")
        print(f"Saved {test_path}")

    if args.save in ("log", "both"):
        train_log = np.log1p(train) / np.log(10.0)
        test_log = np.log1p(test) / np.log(10.0)
        train_log_path = output_dir / "train_log_rain_rate_f4.npy"
        test_log_path = output_dir / "test_log_rain_rate_f4.npy"
        np.save(train_log_path, train_log.astype(np.float32))
        np.save(test_log_path, test_log.astype(np.float32))
        print(f"Saved {train_log_path}")
        print(f"Saved {test_log_path}")


if __name__ == "__main__":
    main()
