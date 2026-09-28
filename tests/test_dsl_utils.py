"""
Tests for DSL parsing and visualization utilities.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from floora.utils.dsl_utils import (
    parse_coord_list,
    parse_holes,
    get_dsl_elements,
    extract_prompt_and_completion,
    visualize_dsl_elements,
    plot_generated_dsl,
)

SAMPLE_DSL_PATH = Path(__file__).parent / "samples" / "dsl_sample.txt"


def test_parse_coord_list():
    assert parse_coord_list("0,0 100, 0 100,100") == [[0, 0], [100, 0], [100, 100]]


def test_parse_holes():
    holes = parse_holes("hole 10,10 20,10 20,20 10,20")
    assert holes == [[[10, 10], [20, 10], [20, 20], [10, 20]]]


def test_parse_holes_ignores_rings_with_too_few_points():
    assert parse_holes("hole 10,10 20,10") == []


def test_get_dsl_elements_empty_string_returns_empty_collections():
    elements = get_dsl_elements("")
    assert elements == {"massing": [], "spaces": [], "building": {}, "structure": {}}


def test_extract_prompt_and_completion_with_both_tags():
    text = (
        "<context> building { occupancy_type residential } </context> "
        "<completion> polygon core 0,0 10,0 10,10 0,10 </completion>"
    )
    prompt_dsl, full_dsl = extract_prompt_and_completion(text)
    assert prompt_dsl == "building { occupancy_type residential }"
    assert full_dsl == (
        "building { occupancy_type residential } polygon core 0,0 10,0 10,10 0,10"
    )


def test_extract_prompt_and_completion_context_only():
    text = "<context> building { occupancy_type residential } </context>"
    prompt_dsl, full_dsl = extract_prompt_and_completion(text)
    assert prompt_dsl == "building { occupancy_type residential }"
    assert full_dsl == prompt_dsl


def test_extract_prompt_and_completion_no_tags_returns_empty():
    prompt_dsl, full_dsl = extract_prompt_and_completion("no tags here")
    assert prompt_dsl == ""
    assert full_dsl == ""


def test_visualize_dsl_elements_draws_massing_and_spaces():
    dsl = (
        "massing { height 10 polygon 0,0 10,0 10,10 0,10 } "
        "polygon core 0,0 5,0 5,5 0,5"
    )
    elements = get_dsl_elements(dsl)
    fig, ax = plt.subplots()
    try:
        visualize_dsl_elements(ax, elements, "Test Title")
        assert ax.get_title() == "Test Title"
        assert len(ax.patches) == 2
        assert len(ax.texts) == 1
    finally:
        plt.close(fig)


def test_plot_generated_dsl_without_labels():
    text = "<context> polygon core 0,0 5,0 5,5 0,5 </context>"
    fig, ax = plt.subplots()
    try:
        plot_generated_dsl(ax, text, "Prompt", show_labels=False)
        assert len(ax.patches) == 1
        assert len(ax.texts) == 0
    finally:
        plt.close(fig)


def test_get_dsl_elements_parses_all_sections():
    generated_text = SAMPLE_DSL_PATH.read_text()

    prompt_dsl, full_dsl = extract_prompt_and_completion(generated_text)
    assert "building {" in prompt_dsl
    assert "spaces {" in full_dsl

    elements = get_dsl_elements(full_dsl)

    assert elements["building"] == {"occupancy_type": "multifamily_residential"}
    assert elements["structure"] == {"material": "reinforced_concrete"}

    assert len(elements["massing"]) == 1
    assert elements["massing"][0]["height"] == 30
    assert len(elements["massing"][0]["coordinates"]) == 6

    space_types = [space["type"] for space in elements["spaces"]]
    assert len(space_types) == 14
    assert space_types.count("core") == 3
    assert space_types.count("corridor") == 1
    assert space_types.count("living_unit") == 10


def test_plot_generated_dsl_real_sample():
    generated_text = SAMPLE_DSL_PATH.read_text()
    fig, ax = plt.subplots()
    try:
        plot_generated_dsl(ax, generated_text, "Real Sample")
        assert len(ax.patches) == 1 + 14  # 1 massing + 14 spaces
    finally:
        plt.close(fig)
