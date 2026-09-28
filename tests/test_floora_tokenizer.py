"""
Tests for the FLOORATokenizer and its registration with transformers.
"""

from pathlib import Path

import pytest
import transformers
from transformers import AutoTokenizer

from floora.tokenizer import FLOORATokenizer
from floora.utils.tokenizer_utils import register_custom_tokenizers

SAMPLE_DSL_PATH = Path(__file__).parent / "samples" / "dsl_sample.txt"

EXPECTED_VOCAB_SIZE = 1367
HUB_REPO_ID = "ADSKAILab/floora-0.6b"


def make_tokenizer(**kwargs):
    return FLOORATokenizer.from_pretrained(**kwargs)


@pytest.fixture
def tokenizer():
    return make_tokenizer()


def test_vocab_size_is_1367(tokenizer):
    assert len(tokenizer) == EXPECTED_VOCAB_SIZE


def test_special_tokens_are_present(tokenizer):
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "</s>"
    assert tokenizer.pad_token == "<pad>"


def test_soc_and_eoc_tokens(tokenizer):
    assert tokenizer.soc_token == "<completion>"
    assert tokenizer.eoc_token == "</completion>"
    assert tokenizer.soc_token_id == tokenizer.convert_tokens_to_ids("<completion>")
    assert tokenizer.eoc_token_id == tokenizer.convert_tokens_to_ids("</completion>")
    assert tokenizer.soc_token_id != tokenizer.unk_token_id
    assert tokenizer.eoc_token_id != tokenizer.unk_token_id


@pytest.mark.parametrize(
    "token",
    [
        "<spec>",
        "<context>",
        "<generate>",
        "<mass>",
        "<grid>",
        "<space>",
        "<structure>",
        "<building>",
        "<column>",
        "<beam>",
        "<core_wall>",
        "<slab>",
        "<outcome>",
        "<unit_mix>",
        "<tool_call>",
        "<tool_response>",
        "<pre>",
        "<suf>",
        "<mid>",
        "polygon",
        "occupancy_type",
        "office_unit",
        "v100",
        "e100",
        "1024",
        "{",
        "}",
        ",",
        ".",
    ],
)
def test_dsl_tokens_are_in_vocab(tokenizer, token):
    assert tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id


def test_add_bos_and_eos_true():
    tok = make_tokenizer(add_bos_token=True, add_eos_token=True)
    ids = tok.encode("polygon")
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id


def test_add_bos_true_eos_false():
    tok = make_tokenizer(add_bos_token=True, add_eos_token=False)
    ids = tok.encode("polygon")
    assert ids[0] == tok.bos_token_id
    assert ids[-1] != tok.eos_token_id


def test_add_bos_false_eos_true():
    tok = make_tokenizer(add_bos_token=False, add_eos_token=True)
    ids = tok.encode("polygon")
    assert ids[0] != tok.bos_token_id
    assert ids[-1] == tok.eos_token_id


def test_add_bos_and_eos_false():
    tok = make_tokenizer(add_bos_token=False, add_eos_token=False)
    ids = tok.encode("polygon")
    assert tok.bos_token_id not in ids
    assert tok.eos_token_id not in ids


def test_call_ignores_add_special_tokens_false(tokenizer):
    forced_true = tokenizer("polygon", add_special_tokens=True)
    forced_false = tokenizer("polygon", add_special_tokens=False)
    assert forced_true["input_ids"] == forced_false["input_ids"]


def test_decode_ignores_skip_special_tokens_true(tokenizer):
    ids = tokenizer.encode("polygon")
    kept = tokenizer.decode(ids, skip_special_tokens=False)
    skipped = tokenizer.decode(ids, skip_special_tokens=True)
    assert kept == skipped
    # default add_bos_token/add_eos_token come from tokenizer_config.json (both True)
    assert tokenizer.bos_token in kept
    assert tokenizer.eos_token in kept


def test_convert_tokens_to_string_handles_none_tokens(tokenizer):
    result = tokenizer.convert_tokens_to_string(["polygon", None, "core"])
    assert tokenizer.unk_token in result


def test_tokenize_dsl_sample_has_no_unk_tokens(tokenizer):
    text = SAMPLE_DSL_PATH.read_text()
    ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    assert tokenizer.unk_token not in tokens


def test_tokenize_dsl_sample_roundtrip(tokenizer):
    text = SAMPLE_DSL_PATH.read_text()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    for expected in ["building", "polygon", "core", "corridor", "living_unit"]:
        assert expected in decoded


def test_save_and_load_pretrained_roundtrip(tmp_path, tokenizer):
    tokenizer.save_pretrained(tmp_path)
    loaded = FLOORATokenizer.from_pretrained(tmp_path)
    assert len(loaded) == len(tokenizer)
    assert loaded.soc_token_id == tokenizer.soc_token_id
    assert loaded.eoc_token_id == tokenizer.eoc_token_id


def test_register_custom_tokenizers_registers_with_auto_mapping():
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    register_custom_tokenizers()
    registered_classes = [
        cls
        for slow, fast in TOKENIZER_MAPPING._extra_content.values()
        for cls in (slow, fast)
        if cls is not None
    ]
    assert FLOORATokenizer in registered_classes


def test_autotokenizer_from_pretrained_hub_repo():
    register_custom_tokenizers()
    try:
        tok = AutoTokenizer.from_pretrained(HUB_REPO_ID, trust_remote_code=True)
    except (OSError, ImportError) as e:
        pytest.skip(
            f"Hub repo {HUB_REPO_ID} not reachable: {e}"
        )

    # trust_remote_code loads a dynamically-imported module, so the resulting
    # class is not the same object as the locally-imported FLOORATokenizer.
    assert type(tok).__name__ == "FLOORATokenizer"
    assert len(tok) == EXPECTED_VOCAB_SIZE
    ids = tok.encode("polygon")
    assert tok.unk_token_id not in ids
