"""
Inference code for FLOORA models.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from floora.inference.config import GenerationConfig
from floora.utils.tokenizer_utils import register_custom_tokenizers

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False


class FlooraEvaluator:
    """Generates completions for FLOORA prompts using HuggingFace or vLLM."""

    def __init__(self, generation_config: Optional[GenerationConfig] = None):
        """Initializes the evaluator and loads the tokenizer and model/engine.

        Args:
            generation_config: Generation and I/O settings. Defaults to
                `GenerationConfig()` if not provided.
        """
        self.config = generation_config or GenerationConfig()
        self.config.print_config()

        if self.config.seed is not None:
            set_seed(self.config.seed)

        tokenizer_name = (
            self.config.tokenizer_name_or_path or self.config.model_name_or_path
        )
        # FLOORATokenizer is bundled with the model repo, so this works without installing this package
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        # Disable eos-appending so prompts end at "</generate>", not at eos
        if hasattr(self.tokenizer, "add_eos_token"):
            self.tokenizer.add_eos_token = False
        self.tokenizer.padding_side = "left"

        # Stop at eos or, if present, the DSL's end-of-completion marker
        self.stop_token_ids = {self.tokenizer.eos_token_id}
        eoc_token_id = getattr(self.tokenizer, "eoc_token_id", None)
        if eoc_token_id is not None:
            self.stop_token_ids.add(eoc_token_id)

        self.model = None
        self.llm = None
        self.device = None
        if self.config.use_vllm:
            self._setup_vllm()
        else:
            self._setup_hf()

        os.makedirs(self.config.output_dir, exist_ok=True)

    def _setup_hf(self) -> None:
        """Loads the model onto `self.device` for HuggingFace-based generation."""
        logger.info("Loading model %s with HuggingFace", self.config.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _setup_vllm(self) -> None:
        """Builds the vLLM engine used for generation.

        Raises:
            ImportError: If vLLM is not installed.
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not available. Install it with `pip install vllm` to use it."
            )
        # Also register locally so vLLM's tokenizer group can resolve FLOORATokenizer by name
        register_custom_tokenizers()
        logger.info("Loading model %s with vLLM", self.config.model_name_or_path)
        self.llm = LLM(
            model=self.config.model_name_or_path,
            tokenizer=self.config.tokenizer_name_or_path
            or self.config.model_name_or_path,
            trust_remote_code=True,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
        )

    def generate_hf(self, prompts: List[str]) -> List[str]:
        """Generates completions for a list of prompts using HuggingFace.

        Args:
            prompts: List of prompt strings to generate completions for.

        Returns:
            List of completion strings, one per prompt, in the same order.
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.stop_token_ids),
            )

        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)
            for ids in output_ids
        ]

    def generate_vllm(self, prompts: List[str]) -> List[str]:
        """Generates completions for a list of prompts using vLLM.

        Args:
            prompts: List of prompt strings to generate completions for.

        Returns:
            List of completion strings, one per prompt, in the same order.
        """
        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            stop_token_ids=list(self.stop_token_ids),
            # vLLM defaults to True, which would strip the DSL's tokens
            skip_special_tokens=False,
        )
        # Tokenize ourselves
        token_prompts = [
            {"prompt_token_ids": self.tokenizer.encode(prompt)} for prompt in prompts
        ]
        outputs = self.llm.generate(token_prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

    def generate(self, prompts: List[str]) -> List[str]:
        """Generates one completion per prompt using the configured backend.

        Args:
            prompts: List of prompt strings to generate completions for.

        Returns:
            List of completion strings, one per prompt, in the same order.
        """
        if self.config.use_vllm:
            return self.generate_vllm(prompts)
        return self.generate_hf(prompts)

    def run(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generates `num_samples` completions for each prompt and saves the results.

        Args:
            prompts: List of prompt strings to generate completions for.

        Returns:
            List of result dicts, one per generated sample.
        """
        results = []
        num_samples = self.config.num_samples
        batch_starts = range(0, len(prompts), self.config.batch_size)
        total_batches = -(-len(prompts) // self.config.batch_size)  # ceil division

        for batch_start in tqdm(
            batch_starts, total=total_batches, desc="Generating completions"
        ):
            batch_prompts = prompts[batch_start : batch_start + self.config.batch_size]
            expanded_prompts = [
                prompt for prompt in batch_prompts for _ in range(num_samples)
            ]

            start_time = time.time()
            completions = self.generate(expanded_prompts)
            elapsed = time.time() - start_time
            logger.info(
                "Generated %d completions in %.2fs (batch starting at prompt %d)",
                len(completions),
                elapsed,
                batch_start,
            )

            for i, (prompt, completion) in enumerate(
                zip(expanded_prompts, completions)
            ):
                results.append(
                    {
                        "prompt_idx": batch_start + i // num_samples,
                        "sample_idx": i % num_samples,
                        "prompt": prompt,
                        "completion": completion,
                        "full_text": prompt + completion,
                    }
                )

        self.save_results(results)
        return results

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Saves generated results to `output_dir/results.json`.

        Args:
            results: List of result dicts, as returned by `run`.
        """
        output_path = os.path.join(self.config.output_dir, "results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved %d results to %s", len(results), output_path)
