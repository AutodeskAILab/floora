"""
Tests for BaseDataset.
"""

import pytest
from datasets import Dataset

from floora.dataset import base_dataset as base_dataset_module
from floora.dataset.base_dataset import BaseDataset


class ConcreteDataset(BaseDataset):
    """Minimal concrete subclass to exercise BaseDataset's non-abstract behavior."""

    def load_data(self):
        return {}

    def format_prompt(self, sample):
        return f"prompt: {sample.get('text', '')}"


def make_dataset():
    return ConcreteDataset("dummy/path", num_workers=1)


def test_base_dataset_is_abstract():
    with pytest.raises(TypeError):
        BaseDataset("dummy/path")


def test_subsample_dataset_with_none_returns_unchanged():
    ds = make_dataset()
    dataset = Dataset.from_dict({"x": list(range(10))})
    result = ds._subsample_dataset(dataset, None, "train")
    assert result is dataset


def test_subsample_dataset_with_fraction():
    ds = make_dataset()
    dataset = Dataset.from_dict({"x": list(range(10))})
    result = ds._subsample_dataset(dataset, 0.5, "train")
    assert len(result) == 5


def test_subsample_dataset_with_int():
    ds = make_dataset()
    dataset = Dataset.from_dict({"x": list(range(10))})
    result = ds._subsample_dataset(dataset, 3, "train")
    assert len(result) == 3


def test_subsample_dataset_int_larger_than_total_is_clamped():
    ds = make_dataset()
    dataset = Dataset.from_dict({"x": list(range(5))})
    result = ds._subsample_dataset(dataset, 100, "train")
    assert len(result) == 5


@pytest.mark.parametrize(
    "nrows",
    [0.0, -0.1, 1.5],
)
def test_subsample_dataset_invalid_fraction_raises(nrows):
    ds = make_dataset()
    dataset = Dataset.from_dict({"x": list(range(10))})
    with pytest.raises(ValueError):
        ds._subsample_dataset(dataset, nrows, "train")


def test_subsample_dataset_invalid_int_raises():
    ds = make_dataset()
    dataset = Dataset.from_dict({"x": list(range(10))})
    with pytest.raises(ValueError):
        ds._subsample_dataset(dataset, 0, "train")


def test_subsample_dataset_invalid_type_raises():
    ds = make_dataset()
    dataset = Dataset.from_dict({"x": list(range(10))})
    with pytest.raises(TypeError):
        ds._subsample_dataset(dataset, "5", "train")


def test_load_raw_dataset_without_config(monkeypatch):
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"train": "dummy_train"}

    monkeypatch.setattr(base_dataset_module, "load_dataset", fake_load_dataset)
    ds = ConcreteDataset("some/path", num_workers=1)
    result = ds._load_raw_dataset()
    assert result == {"train": "dummy_train"}
    assert captured["args"] == ("some/path",)


def test_load_raw_dataset_with_config(monkeypatch):
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        return {"train": "dummy_train"}

    monkeypatch.setattr(base_dataset_module, "load_dataset", fake_load_dataset)
    ds = ConcreteDataset("some/path", dataset_config="subset", num_workers=1)
    ds._load_raw_dataset()
    assert captured["args"] == ("some/path", "subset")


def test_load_and_subsample_dataset_only_processes_existing_splits(monkeypatch):
    ds = ConcreteDataset("some/path", train_nrows=2, eval_nrows=0.5, num_workers=1)
    raw = {"train": Dataset.from_dict({"x": list(range(10))})}
    monkeypatch.setattr(ds, "_load_raw_dataset", lambda: raw)

    result = ds._load_and_subsample_dataset()

    assert "eval" not in result
    assert len(result["train"]) == 2


def test_add_prompts_to_dataset():
    ds = make_dataset()
    dataset = Dataset.from_dict({"text": ["a", "b", "c"]})
    result = ds._add_prompts_to_dataset(dataset)
    assert result["prompt"] == ["prompt: a", "prompt: b", "prompt: c"]
