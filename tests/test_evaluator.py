"""
Tests for FlooraEvaluator's HuggingFace and vLLM inference backends.

These use the real FLOORATokenizer and a tiny real causal LM from the Hub
(trl-internal-testing/tiny-Qwen3ForCausalLM) instead of mocks, so they
exercise the actual tokenization/generation code paths.
"""

import json

import pytest

from floora.inference import evaluator as evaluator_module
from floora.inference.config import GenerationConfig
from floora.inference.evaluator import FlooraEvaluator
from floora.tokenizer import FLOORATokenizer

TINY_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"

PROMPT = "<context> <building> building { occupancy_type residential } </building> </context> <generate> <space> </generate>"


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    """FLOORATokenizer, saved to a fresh, self-contained directory."""
    tokenizer = FLOORATokenizer.from_pretrained()
    tokenizer.register_for_auto_class("AutoTokenizer")
    save_dir = tmp_path_factory.mktemp("floora_tokenizer")
    tokenizer.save_pretrained(save_dir)
    return str(save_dir)


def make_evaluator(tmp_path, tokenizer_dir, **config_kwargs):
    config = GenerationConfig(
        model_name_or_path=TINY_MODEL,
        tokenizer_name_or_path=tokenizer_dir,
        output_dir=str(tmp_path),
        max_new_tokens=8,
        **config_kwargs,
    )
    try:
        return FlooraEvaluator(config)
    except OSError as e:
        pytest.skip(f"Hub repo {TINY_MODEL} not reachable: {e}")


class TestHuggingFaceBackend:
    def test_setup_hf_loads_model_and_tokenizer(self, tmp_path, tokenizer_dir):
        evaluator = make_evaluator(tmp_path, tokenizer_dir)
        assert evaluator.model is not None
        assert evaluator.llm is None
        assert evaluator.tokenizer.padding_side == "left"
        assert evaluator.tokenizer.add_eos_token is False
        assert evaluator.stop_token_ids == {
            evaluator.tokenizer.eos_token_id,
            evaluator.tokenizer.eoc_token_id,
        }

    def test_generate_hf_returns_one_completion_per_prompt(
        self, tmp_path, tokenizer_dir
    ):
        evaluator = make_evaluator(tmp_path, tokenizer_dir)
        completions = evaluator.generate_hf([PROMPT, PROMPT])
        assert len(completions) == 2
        assert all(isinstance(completion, str) for completion in completions)

    def test_generate_dispatches_to_hf_when_not_use_vllm(self, tmp_path, tokenizer_dir):
        evaluator = make_evaluator(tmp_path, tokenizer_dir)
        completions = evaluator.generate([PROMPT, PROMPT])
        assert len(completions) == 2

    def test_run_expands_prompts_by_num_samples_and_saves_results(
        self, tmp_path, tokenizer_dir
    ):
        evaluator = make_evaluator(tmp_path, tokenizer_dir, num_samples=2, batch_size=2)
        prompts = [PROMPT, PROMPT, PROMPT]

        results = evaluator.run(prompts)

        assert len(results) == len(prompts) * 2
        for prompt_idx, prompt in enumerate(prompts):
            for sample_idx in range(2):
                matches = [
                    r
                    for r in results
                    if r["prompt_idx"] == prompt_idx and r["sample_idx"] == sample_idx
                ]
                assert len(matches) == 1
                assert matches[0]["prompt"] == prompt
                assert matches[0]["full_text"] == prompt + matches[0]["completion"]

        output_path = tmp_path / "results.json"
        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == results


class TestVLLMBackend:
    """These tests only run if vllm is actually installed."""

    @pytest.fixture(autouse=True)
    def skip_if_vllm_unavailable(self):
        if not evaluator_module.VLLM_AVAILABLE:
            pytest.skip("vllm is not installed")

    def test_setup_vllm_builds_llm_with_trust_remote_code(
        self, tmp_path, tokenizer_dir
    ):
        evaluator = make_evaluator(tmp_path, tokenizer_dir, use_vllm=True)
        assert evaluator.model is None
        assert evaluator.llm is not None

    def test_generate_vllm_returns_one_completion_per_prompt(
        self, tmp_path, tokenizer_dir
    ):
        evaluator = make_evaluator(tmp_path, tokenizer_dir, use_vllm=True)
        completions = evaluator.generate_vllm([PROMPT, PROMPT])
        assert len(completions) == 2
        assert all(isinstance(completion, str) for completion in completions)

    def test_generate_dispatches_to_vllm_when_use_vllm(self, tmp_path, tokenizer_dir):
        evaluator = make_evaluator(tmp_path, tokenizer_dir, use_vllm=True)
        completions = evaluator.generate([PROMPT, PROMPT])
        assert len(completions) == 2

    def test_run_saves_results_with_vllm_backend(self, tmp_path, tokenizer_dir):
        evaluator = make_evaluator(
            tmp_path, tokenizer_dir, use_vllm=True, num_samples=1
        )
        results = evaluator.run([PROMPT, PROMPT])

        assert len(results) == 2
        assert (tmp_path / "results.json").exists()


def test_setup_vllm_raises_when_vllm_unavailable(tmp_path, tokenizer_dir, monkeypatch):
    """Doesn't require vllm to be installed: forces the unavailable path."""
    monkeypatch.setattr(evaluator_module, "VLLM_AVAILABLE", False)
    with pytest.raises(ImportError):
        make_evaluator(tmp_path, tokenizer_dir, use_vllm=True)
