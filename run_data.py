"""Example script to run the DSLDataset for a specific dataset."""

import argparse
import os
from datetime import datetime

from matplotlib import pyplot as plt

from floora.dataset import DSLDataset
from floora.utils import plot_generated_dsl


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run DSLDataset for a specific dataset"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ADSKAILab/floora_dataset",
        help="Dataset name or path. Default: ADSKAILab/floora_dataset",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="synthetic",
        help="Dataset config name to use. Default: synthetic",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to display/plot. Default: test",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to display. Default: 10",
    )

    # Subsampling parameters
    parser.add_argument(
        "--train_nrows",
        type=str,
        default=None,
        help="Number of training rows to use. If float (0-1), treats as fraction. "
        "If int, selects that many random samples. Default: None (use all data)",
    )
    parser.add_argument(
        "--eval_nrows",
        type=str,
        default=None,
        help="Number of evaluation rows to use. If float (0-1), treats as fraction. "
        "If int, selects that many random samples. Default: None (use all data)",
    )

    # Space sampling parameters
    parser.add_argument(
        "--probability_of_space_sampling",
        type=float,
        default=0.5,
        help="Probability of sampling spaces in the prompt. Default: 0.5",
    )
    parser.add_argument(
        "--min_space_sampling_ratio",
        type=float,
        default=0.1,
        help="Minimum ratio of spaces to sample in the prompt. Default: 0.1",
    )
    parser.add_argument(
        "--max_space_sampling_ratio",
        type=float,
        default=0.6,
        help="Maximum ratio of spaces to sample in the prompt. Default: 0.6",
    )

    # Plotting parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save sample plots to. "
        "Default: plots_<dataset>_<config>_<timestamp>",
    )

    return parser.parse_args()


def parse_nrows(nrows_str):
    """
    Parse nrows argument from string to int or float.

    Args:
        nrows_str: String representation of nrows

    Returns:
        int, float, or None
    """
    if nrows_str is None:
        return None

    try:
        # Try to parse as float first
        value = float(nrows_str)

        # If it's a whole number, convert to int
        if value.is_integer() and value > 1:
            return int(value)

        # Otherwise return as float (for fractions between 0-1)
        return value
    except ValueError as exc:
        raise ValueError(
            f"Invalid nrows value: {nrows_str}. Must be a number."
        ) from exc


def short_name(name: str) -> str:
    """Strips a HuggingFace Hub org/user prefix (e.g. "ADSKAILab/") from a repo id."""
    return name.rstrip("/").split("/")[-1]


def main():
    """
    Main function to run the DSL dataset example.
    """
    args = parse_args()

    # Parse nrows arguments
    train_nrows = parse_nrows(args.train_nrows)
    eval_nrows = parse_nrows(args.eval_nrows)

    dataset = DSLDataset(
        dataset_name_or_path=args.dataset_name,
        dataset_config=args.config_name,
        context_fields=["dsl_building", "dsl_structure", "dsl_mass", "dsl_spaces"],
        target_fields=["dsl_spaces"],
        probability_of_space_sampling=args.probability_of_space_sampling,
        min_space_sampling_ratio=args.min_space_sampling_ratio,
        max_space_sampling_ratio=args.max_space_sampling_ratio,
        train_nrows=train_nrows,
        eval_nrows=eval_nrows,
    )

    datasets = dataset.load_data()
    split_data = datasets[args.split]

    print(
        f"\nLoaded {len(split_data)} samples from {args.dataset_name} ({args.split} split)"
    )
    nrows = train_nrows if args.split == "train" else eval_nrows
    if nrows is not None:
        if isinstance(nrows, float):
            print(f"  (subsampled to {nrows*100:.1f}% of {args.split} data)")
        else:
            print(f"  (subsampled to {nrows} {args.split} samples)")
    print()

    output_dir = args.output_dir or (
        f"plots_{short_name(args.dataset_name)}_{args.config_name}"
        f"_{datetime.now():%Y%m%d_%H%M%S}"
    )
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(
        split_data.select(range(min(args.num_samples, len(split_data))))
    ):
        print(f"Sample {i+1}:")
        print(sample["prompt"])
        print("...\n")

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_generated_dsl(ax, sample["prompt"], title=f"Sample {i+1}")
        fig.savefig(os.path.join(output_dir, f"sample_{i+1}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved {min(args.num_samples, len(split_data))} plots to {output_dir}/")


if __name__ == "__main__":
    main()
