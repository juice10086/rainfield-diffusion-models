from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description="Generate simulated OpenMRG CML links.")
    parser.add_argument("--n-links", type=int, required=True, help="Number of links.")
    parser.add_argument(
        "--max-len",
        type=float,
        required=True,
        help="Maximum link length in grid units.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=321,
        help="Random seed used for link generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp_settings"),
        help="Directory where generated links are saved.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for generation, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=48,
        help="Rain-field grid height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=36,
        help="Rain-field grid width.",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=1.0,
        help="Minimum link length in grid units.",
    )
    parser.add_argument(
        "--n-search-iter",
        type=int,
        default=1000,
        help="Maximum number of random-search iterations.",
    )
    args = parser.parse_args()

    import torch

    from dm_ps.inv_prob.create_inv_prob_openmrg import generate_random_links

    torch.set_default_device(args.device)
    torch.manual_seed(args.seed)

    links = generate_random_links(
        n_links=args.n_links,
        height=args.height,
        width=args.width,
        min_length=args.min_length,
        max_length=args.max_len,
        n_search_iter=args.n_search_iter,
        device=args.device,
    )
    if len(links) != args.n_links:
        raise RuntimeError(
            f"Generated {len(links)} links, but --n-links requested {args.n_links}. "
            f"Consider increasing the `n_search_iter`."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_len_label = f"{args.max_len:g}"
    output_path = (
        args.output_dir
        / f"simulated_seed_{args.seed}_n_links_{args.n_links}_max_len_{max_len_label}.pt"
    )
    torch.save(links, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
