# FLOORA: Floor Layout Optimization with RL Alignment

Code accompanying the FLOORA paper: a DSL tokenizer, dataset loader, and inference pipeline for generating architectural floor plans with a FLOORA language model.

- Dataset: [ADSKAILab/floora_dataset](https://huggingface.co/datasets/ADSKAILab/floora_dataset)
- Models: [ADSKAILab/floora-0.6b](https://huggingface.co/ADSKAILab/floora-0.6b), [ADSKAILab/floora-1.7b](https://huggingface.co/ADSKAILab/floora-1.7b)

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

To also run inference with vLLM (in addition to HuggingFace), install the
`vllm` extra:

```bash
uv sync --extra vllm
```

## Usage

### Visualize the dataset

[run_data.py](run_data.py) loads the dataset, builds prompts from its DSL
fields, and plots a few samples:

```bash
uv run python run_data.py --config_name synthetic --num_samples 10
```

Key arguments:
- `--dataset_name`: Hub dataset repo. Default: `ADSKAILab/floora_dataset`.
- `--config_name`: dataset config/subset (e.g. `synthetic`, `osm`).
- `--num_samples`: number of samples to plot.
- `--train_nrows` / `--eval_nrows`: subsample the train/test split (int count or 0-1 fraction).
- `--output_dir`: defaults to `plots_<dataset>_<config>_<timestamp>`.

### Run inference

[run_inference.py](run_inference.py) loads prompts from the dataset, generates
completions with a FLOORA model (HuggingFace or vLLM), saves the raw results,
and plots each generated design:

```bash
uv run python run_inference.py --model_name_or_path ADSKAILab/floora-0.6b --num_prompts 10
```

Use `--use_vllm` to generate with vLLM instead of HuggingFace (requires the
`vllm` extra), and `--num_samples` to draw multiple completions per prompt
("k"):

```bash
uv run python run_inference.py --use_vllm --num_samples 5
```

Key arguments:
- `--dataset_name` / `--config_name` / `--split`: dataset to pull prompts from (`--split` is `train` or `test`).
- `--num_prompts`: number of prompts to run inference on.
- `--model_name_or_path` / `--tokenizer_name_or_path`: Hub repos to load (tokenizer defaults to the model repo).
- `--num_samples`: completions to sample per prompt (k); each is saved and plotted individually.
- `--batch_size`, `--max_new_tokens`, `--temperature`, `--top_k`, `--top_p`, `--seed`: generation settings.
- `--output_dir`: defaults to `results_<model>_<dataset>_<config>_<split>_<timestamp>`.

Results are saved as `results.json` (one entry per generated sample) alongside
a `.png` plot per prompt/sample.

### Tests

```bash
uv run pytest tests
```

## Project structure

```
floora/
├── run_data.py               # Visualize dataset samples
├── run_inference.py          # Run HF/vLLM inference and plot results
├── src/floora/
│   ├── constants.py          # DSL tag/color mappings
│   ├── dataset/
│   │   ├── base_dataset.py   # Abstract HF dataset loader (splits, subsampling)
│   │   └── dsl_dataset.py    # DSL prompt templating and dataset loading
│   ├── inference/
│   │   ├── config.py         # GenerationConfig
│   │   └── evaluator.py      # FlooraEvaluator: HF/vLLM generation
│   ├── tokenizer/
│   │   ├── floora_tokenizer.py   # FLOORATokenizer (GPT2-based, DSL vocab)
│   │   └── tokenizer_files/      # Bundled vocab/config used as the default
│   └── utils/
│       ├── dsl_utils.py          # DSL parsing and plotting
│       └── tokenizer_utils.py    # AutoTokenizer registration helper
└── tests/                    # Unit tests for the above
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
