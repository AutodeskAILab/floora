"""
Example script to run FLOORA inference and plot the generated designs.
"""

import argparse
import os
from datetime import datetime

from matplotlib import pyplot as plt
from tqdm import tqdm

from floora.dataset import DSLDataset
from floora.inference import GenerationConfig, FlooraEvaluator
from floora.utils import plot_generated_dsl


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run FLOORA inference on a dataset and plot the results"
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
        help="Dataset split to run inference on. Default: test",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=10,
        help="Number of prompts to run inference on. Default: 10",
    )

    # Model parameters
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ADSKAILab/floora-0.6b",
        help="HuggingFace Hub model repo to load. Default: ADSKAILab/floora-0.6b",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="HuggingFace Hub tokenizer repo. Defaults to --model_name_or_path.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=False,
        help="Use vLLM instead of HuggingFace for generation.",
    )

    # Generation parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of completions to sample per prompt (k). Default: 1",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results and plots to. "
        "Default: results_<model>_<dataset>_<config>_<split>_<timestamp>",
    )

    return parser.parse_args()


def short_name(name: str) -> str:
    """Strips a HuggingFace Hub org/user prefix (e.g. "ADSKAILab/") from a repo id."""
    return name.rstrip("/").split("/")[-1]


def main():
    """
    Main function to run FLOORA inference and plot the generated designs.
    """
    args = parse_args()

    output_dir = args.output_dir or (
        f"results_{short_name(args.model_name_or_path)}"
        f"_{short_name(args.dataset_name)}_{args.config_name}_{args.split}"
        f"_{datetime.now():%Y%m%d_%H%M%S}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Only subsample the split we're actually going to run inference on.
    nrows_kwarg = {
        "train": "train_nrows",
        "test": "eval_nrows",
    }[args.split]

    dataset = DSLDataset(
        dataset_name_or_path=args.dataset_name,
        dataset_config=args.config_name,
        context_fields=["dsl_building", "dsl_structure", "dsl_mass", "dsl_spaces"],
        target_fields=["dsl_spaces"],
        **{nrows_kwarg: args.num_prompts},
    )
    datasets = dataset.load_data()
    split_data = datasets[args.split]
    prompts = split_data.select(range(min(args.num_prompts, len(split_data))))["prompt"]

    generation_config = GenerationConfig(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        output_dir=output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_vllm=args.use_vllm,
        seed=args.seed,
    )

    evaluator = FlooraEvaluator(generation_config)
    results = evaluator.run(prompts)

    print(
        f"\nGenerated {len(results)} completion(s) for {len(prompts)} prompt(s) "
        f"-> {output_dir}/results.json"
    )

    for result in tqdm(results, desc="Plotting results"):
        fig, ax = plt.subplots(figsize=(6, 6))
        title = f"prompt {result['prompt_idx']} / sample {result['sample_idx']}"
        plot_generated_dsl(ax, result["full_text"], title=title)
        plot_path = os.path.join(
            output_dir, f"prompt{result['prompt_idx']}_sample{result['sample_idx']}.png"
        )
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    print(f"Saved {len(results)} plots to {output_dir}/")


if __name__ == "__main__":
    main()
