"""
DSL-related constants and mappings.
"""

from typing import Dict

# DSL data column to DSL field mapping
DSL_DATA_COLUMN_TO_DSL_TAG = {
    "dsl_mass": "mass",
    "dsl_structure": "structure",
    "dsl_building": "building",
    "dsl_spaces": "space",
}

# DSL field to DSL data column mapping (automatically reversed)
DSL_TAG_TO_DSL_DATA_COLUMN = {v: k for k, v in DSL_DATA_COLUMN_TO_DSL_TAG.items()}

# DSL tags to DSL field mapping
DSL_TAG_TO_DSL_FIELD = {
    "mass": "massing",
    "space": "spaces",
    "building": "building",
    "structure": "structure",
}

# Palette for DSL space types
SPACE_COLORS: Dict[str, str] = {
    "core": "#5E89BC",
    "corridor": "#E3B36B",
    "living_unit": "#83B383",
    "undefined": "#C0C4C9",
}
DEFAULT_SPACE_COLOR = "#C0C4C9"

# Colors for non-space geometry.
MASSING_FACECOLOR = "#E9E9EC"
MASSING_EDGECOLOR = "#5A5A5A"
