from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def main():
    parser = ArgumentParser(
        description="Compute OpenMRG CML power-law constants from metadata."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/cml_metadata.npz"),
        help="Path to the CML metadata NPZ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where cst_a_power_b_openmrg.pt is saved.",
    )
    args = parser.parse_args()

    import torch
    from pynncml.single_cml_methods.power_law.pl_module import a_b_parameters

    metadata = np.load(args.metadata_path)
    cmls_a_and_b = torch.tensor(
        [
            [el.item() for el in a_b_parameters(frequency, polarization)]
            for frequency, polarization in zip(
                metadata["frequency"], metadata["polarization"]
            )
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "cst_a_power_b_openmrg.pt"
    torch.save(cmls_a_and_b.T, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
