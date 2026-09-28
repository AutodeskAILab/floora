"""
Utility functions for parsing and visualizing FLOORA DSL elements.
"""

from typing import Dict, Tuple, Any, List

import re
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator

from floora.constants import (
    SPACE_COLORS,
    DEFAULT_SPACE_COLOR,
    MASSING_FACECOLOR,
    MASSING_EDGECOLOR,
)


def parse_coord_list(coords_text: str) -> List[List[int]]:
    """
    Parse a coordinate text string into a list of [x, y] integer pairs.

    Args:
        coords_text: String of whitespace-separated "x,y" coordinate pairs

    Returns:
        List of [x, y] integer pairs
    """
    points = []
    for coord_pair in re.findall(r"\d+\s*,\s*\d+", coords_text):
        points.append(list(map(int, re.split(r"\s*,\s*", coord_pair))))
    return points


def parse_holes(holes_block_text: str) -> List[List[List[int]]]:
    """
    Parse the content of a holes { ... } block into a list of hole rings.

    Args:
        holes_block_text: Content of a holes { ... } block, e.g.
            "hole 10,10 20,10 20,20 10,20"

    Returns:
        List of hole rings, each a list of [x, y] pairs
    """
    hole_rings = []
    for match in re.finditer(r"hole\s+((?:\d+\s*,\s*\d+\s*)+)", holes_block_text):
        ring = parse_coord_list(match.group(1))
        if len(ring) >= 3:
            hole_rings.append(ring)
    return hole_rings


def get_dsl_elements(dsl_string: str) -> Dict[str, Any]:
    """
    Parse a DSL string and extract its elements for verification.

    Args:
        dsl_string: Raw DSL text containing building, structure, massing,
            and space definitions

    Returns:
        Dict with "building", "structure", "massing", and "spaces" keys
    """
    elements = {
        "massing": [],
        "spaces": [],
        "building": {},
        "structure": {},
    }

    # Parse building
    # Pattern: building { occupancy_type <type> storeys <num> level <num> elevation <num> }
    building_pattern = r"building\s*\{([^}]+)\}"
    building_match = re.search(building_pattern, dsl_string)
    if building_match:
        building_content = building_match.group(1)

        # Extract occupancy_type
        occupancy_match = re.search(r"occupancy_type\s+(\w+)", building_content)
        if occupancy_match:
            elements["building"]["occupancy_type"] = occupancy_match.group(1)

    # Parse structure
    # Pattern: structure { material <material_type> }
    structure_pattern = r"structure\s*\{\s*material\s+(\w+)\s*\}"
    structure_match = re.search(structure_pattern, dsl_string)
    if structure_match:
        elements["structure"]["material"] = structure_match.group(1)

    # Parse massing
    # Pattern: massing { height <height> polygon <coords> [holes { hole <coords> ... }] }
    # Uses a block-level match to handle the optional nested holes { } brace.
    massing_block_pattern = r"massing\s*\{([^{}]*(?:\{[^}]*\}[^{}]*)*)\}"
    for block_match in re.finditer(massing_block_pattern, dsl_string):
        block_content = block_match.group(1)
        height_match = re.search(r"height\s+(\d+)", block_content)
        polygon_match = re.search(r"polygon\s+((?:\d+\s*,\s*\d+\s*)+)", block_content)
        if not height_match or not polygon_match:
            continue
        height = int(height_match.group(1))
        outer_points = parse_coord_list(polygon_match.group(1))
        holes_block_match = re.search(r"holes\s*\{([^}]*)\}", block_content)
        hole_rings = (
            parse_holes(holes_block_match.group(1)) if holes_block_match else []
        )
        elements["massing"].append(
            {"height": height, "coordinates": outer_points, "holes": hole_rings}
        )

    # Parse spaces
    # Pattern: polygon <type> <coords> [holes { hole <coords> ... }]
    space_pattern = (
        r"polygon\s+(\w+)\s+((?:\d+\s*,\s*\d+\s*)+)(?:\s*holes\s*\{([^}]*)\})?"
    )
    for match in re.finditer(space_pattern, dsl_string):
        space_type = match.group(1)
        outer_points = parse_coord_list(match.group(2))
        hole_rings = parse_holes(match.group(3)) if match.group(3) else []
        elements["spaces"].append(
            {"type": space_type, "coordinates": outer_points, "holes": hole_rings}
        )

    return elements


def extract_prompt_and_completion(generated_text: str) -> Tuple[str, str]:
    """
    Extract prompt DSL and completion DSL from generated text.

    Args:
        generated_text: Text containing <context> and optionally <completion> tags

    Returns:
        Tuple of (prompt_dsl, full_dsl), where prompt_dsl is the content within
        <context> tags only, and full_dsl combines the <context> and
        <completion> tag contents
    """
    dsl_parts = []
    prompt_dsl = ""

    # First extract content from context tags (this is the prompt)
    context_pattern = r"<context>(.*?)</context>"
    context_match = re.search(context_pattern, generated_text, re.DOTALL)
    if context_match:
        prompt_dsl = context_match.group(1).strip()
        if prompt_dsl:
            dsl_parts.append(prompt_dsl)

    # Then extract content from completion tags
    completion_pattern = r"<completion>(.*?)(?:</completion>|$)"
    completion_match = re.search(completion_pattern, generated_text, re.DOTALL)
    if completion_match:
        completion_content = completion_match.group(1).strip()
        if completion_content:
            dsl_parts.append(completion_content)

    # Stitch the parts together for full DSL
    full_dsl = " ".join(dsl_parts) if dsl_parts else ""

    return prompt_dsl, full_dsl


def plot_generated_dsl(ax, generated_text: str, title: str, show_labels: bool = True):
    """
    Extract, parse, and visualize DSL elements from generated text on a matplotlib axis.

    Works for prompt-only text (a <context> tag), completion-only text, or the
    combination of both, since <context> and <completion> content are stitched
    together before parsing.

    Args:
        ax: Matplotlib axis to draw on
        generated_text: Text containing a <context> tag, a <completion> tag, or both
        title: Title to set on the axis
        show_labels: Whether to label each space polygon with its space type

    Returns:
        None
    """
    _, full_dsl = extract_prompt_and_completion(generated_text)
    elements = get_dsl_elements(full_dsl)
    visualize_dsl_elements(ax, elements, title, show_labels=show_labels)


def visualize_dsl_elements(ax, elements: Dict, title: str, show_labels: bool = True):
    """
    Visualize DSL elements on a matplotlib axis.

    Args:
        ax: Matplotlib axis to draw on
        elements: Parsed DSL elements, as returned by get_dsl_elements
        title: Title to set on the axis
        show_labels: Whether to label each space polygon with its space type

    Returns:
        None
    """
    # Calculate bounds
    all_x = []
    all_y = []

    # Add massing
    for massing in elements.get("massing", []):
        coords = massing["coordinates"]
        if coords:
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            all_x.extend(x_coords)
            all_y.extend(y_coords)

            # Draw massing as a soft gray polygon
            polygon = patches.Polygon(
                coords,
                linewidth=1.5,
                edgecolor=MASSING_EDGECOLOR,
                facecolor=MASSING_FACECOLOR,
                alpha=0.45,
                zorder=1,
            )
            ax.add_patch(polygon)

    # Add spaces
    for space in elements.get("spaces", []):
        coords = space["coordinates"]
        space_type = space["type"]
        if coords:
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            all_x.extend(x_coords)
            all_y.extend(y_coords)

            # Get color for space type
            color = SPACE_COLORS.get(space_type, DEFAULT_SPACE_COLOR)

            # Draw space polygon
            polygon = patches.Polygon(
                coords,
                linewidth=1.0,
                edgecolor="0.4",
                facecolor=color,
                alpha=0.65,
                zorder=3,
            )
            ax.add_patch(polygon)

            # Add label if requested
            if show_labels and len(coords) >= 3:
                # Calculate centroid
                centroid_x = sum(x_coords) / len(x_coords)
                centroid_y = sum(y_coords) / len(y_coords)

                label = space_type
                ax.text(
                    centroid_x,
                    centroid_y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="0.15",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.8,
                    ),
                    zorder=4,
                )

    # Set axis properties
    if all_x and all_y:
        padding = 20
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    else:
        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 110)

    ax.set_aspect("equal")

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("0.4")
    ax.tick_params(colors="0.4", length=4)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="0.88", linewidth=0.8)
    # Dotted minor grid on the y-axis only; keep fixed model-size x-ticks clean.
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="minor", axis="y", color="0.9", linewidth=0.6, linestyle=":")
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=11, fontweight="normal", pad=8)

    # Keep coordinate tick labels readable but unobtrusive.
    ax.tick_params(axis="both", which="major", labelsize=7)
