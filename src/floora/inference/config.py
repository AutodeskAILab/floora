"""
Configuration for FLOORA inference.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ADSKAILab/floora-0.6b"


@dataclass
class GenerationConfig:
    """Configuration for generating completions with a FLOORA model.

    Attributes:
        model_name_or_path: HuggingFace Hub repo id (or local path) of the model.
        tokenizer_name_or_path: HuggingFace Hub repo id (or local path) of the
            tokenizer. Defaults to `model_name_or_path` when unset, since the
            tokenizer is hosted alongside the model.
        output_dir: Directory results are saved to.
        num_samples: Number of completions to sample per prompt ("k"); no
            pass@k scoring is done.
        batch_size: Number of prompts generated per batch.
        max_new_tokens: Maximum number of tokens to generate per completion.
        temperature: Sampling temperature.
        top_k: Top-k sampling cutoff.
        top_p: Top-p (nucleus) sampling cutoff.
        use_vllm: Whether to generate with vLLM instead of HuggingFace.
        vllm_gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        seed: Random seed for reproducibility. `None` disables seeding.
    """

    model_name_or_path: str = DEFAULT_MODEL
    # Defaults to model_name_or_path when unset
    tokenizer_name_or_path: Optional[str] = None
    output_dir: str = "results"
    # Number of completions to sample per prompt ("k")
    num_samples: int = 1
    batch_size: int = 8
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_k: int = 30
    top_p: float = 0.9
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.9
    seed: Optional[int] = 42

    def __post_init__(self):
        """Validates configuration values after initialization.

        Raises:
            AssertionError: If any field holds an invalid value.
        """
        assert self.max_new_tokens > 0, "max_new_tokens must be positive"
        assert 0 <= self.temperature <= 5.0, "temperature must be between 0 and 5.0"
        assert self.top_k >= 0, "top_k must be non-negative"
        assert 0 <= self.top_p <= 1.0, "top_p must be between 0 and 1.0"
        assert self.num_samples > 0, "num_samples must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert (
            0 < self.vllm_gpu_memory_utilization <= 1.0
        ), "vllm_gpu_memory_utilization must be between 0 and 1.0"
        assert (
            self.seed is None or self.seed >= 0
        ), "seed must be non-negative if specified"

    def print_config(self):
        """Logs the generation configuration."""
        logger.info("Generation configuration:")
        logger.info("  Model: %s", self.model_name_or_path)
        logger.info("  Backend: %s", "vLLM" if self.use_vllm else "HuggingFace")
        logger.info("  Samples per prompt (k): %d", self.num_samples)
        logger.info("  Temperature: %.2f", self.temperature)
        logger.info("  Max new tokens: %d", self.max_new_tokens)
