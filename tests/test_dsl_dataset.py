"""
Tests for DSLPromptTemplate and DSLDataset.
"""

import pytest
from datasets import Dataset

from floora.dataset import dsl_dataset as dsl_dataset_module
from floora.dataset.dsl_dataset import DSLDataset, DSLPromptTemplate

SPACES_STR = (
    "spaces { "
    "polygon core 0,0 100,0 100,100 0,100 "
    "polygon undefined 100,0 150,0 150,100 100,100 "
    "}"
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probability_of_space_sampling": 1.5},
        {"min_space_sampling_ratio": -0.1},
        {"max_space_sampling_ratio": 1.1},
        {"min_space_sampling_ratio": 0.8, "max_space_sampling_ratio": 0.2},
    ],
)
def test_prompt_template_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        DSLPromptTemplate(
            context_fields=["dsl_spaces"], target_fields=["dsl_spaces"], **kwargs
        )


def test_build_target_template():
    template = DSLPromptTemplate(
        context_fields=["dsl_building"], target_fields=["dsl_spaces"]
    )
    assert template.target_template == "<generate> <space> </generate>"


def test_extract_spaces_splits_polygons():
    template = DSLPromptTemplate(
        context_fields=["dsl_spaces"], target_fields=["dsl_spaces"]
    )
    spaces = template._extract_spaces(SPACES_STR)
    assert spaces == [
        "polygon core 0,0 100,0 100,100 0,100",
        "polygon undefined 100,0 150,0 150,100 100,100",
    ]


def test_sample_spaces_returns_none_when_field_not_in_context():
    template = DSLPromptTemplate(
        context_fields=["dsl_building"], target_fields=["dsl_spaces"]
    )
    assert template._sample_spaces(SPACES_STR) is None


def test_sample_spaces_excludes_undefined(monkeypatch):
    template = DSLPromptTemplate(
        context_fields=["dsl_spaces"],
        target_fields=["dsl_spaces"],
        min_space_sampling_ratio=1.0,
        max_space_sampling_ratio=1.0,
    )
    monkeypatch.setattr(dsl_dataset_module.random, "uniform", lambda a, b: 1.0)
    monkeypatch.setattr(
        dsl_dataset_module.random, "sample", lambda population, k: population[:k]
    )
    result = template._sample_spaces(SPACES_STR)
    assert result == "spaces { polygon core 0,0 100,0 100,100 0,100 }"


def test_format_prompt_deterministic(monkeypatch):
    monkeypatch.setattr(
        dsl_dataset_module.random, "sample", lambda population, k: list(population)[:k]
    )
    monkeypatch.setattr(dsl_dataset_module.random, "uniform", lambda a, b: 1.0)

    template = DSLPromptTemplate(
        context_fields=["dsl_building", "dsl_spaces"],
        target_fields=["dsl_spaces"],
        probability_of_space_sampling=1.0,
        min_space_sampling_ratio=1.0,
        max_space_sampling_ratio=1.0,
    )
    sample = {
        "dsl_building": "building { occupancy_type residential }",
        "dsl_spaces": SPACES_STR,
    }

    prompt = template.format_prompt(sample)

    expected = (
        "<context> "
        "<building> building { occupancy_type residential } </building> "
        "<space> spaces { polygon core 0,0 100,0 100,100 0,100 } </space>"
        " </context> <generate> <space> </generate>"
    )
    assert prompt == expected


def test_dsl_dataset_load_data_adds_prompts(monkeypatch):
    ds = DSLDataset(
        "dummy/path",
        context_fields=["dsl_building"],
        target_fields=["dsl_spaces"],
        num_workers=1,
    )
    raw = {
        "train": Dataset.from_dict(
            {"dsl_building": ["building { occupancy_type residential }"]}
        ),
        "test": Dataset.from_dict(
            {"dsl_building": ["building { occupancy_type office }"]}
        ),
    }
    monkeypatch.setattr(ds, "_load_and_subsample_dataset", lambda: raw)

    datasets = ds.load_data()

    assert "prompt" in datasets["train"].column_names
    assert "prompt" in datasets["test"].column_names
    assert ds.train_dataset is datasets["train"]
    assert ds.eval_dataset is datasets["test"]
