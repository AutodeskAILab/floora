"""
FLOORA-specific data classes.
"""

from typing import Dict, Any, List, Optional
import random
import logging
import re

from datasets import Dataset

from floora.dataset.base_dataset import BaseDataset
from floora.constants import DSL_DATA_COLUMN_TO_DSL_TAG

logger = logging.getLogger(__name__)

# Captures each polygon entry together with its optional trailing holes { ... } block
_RE_DSL_POLYGON = re.compile(
    r"(polygon\s+\w+\s+.*?)(?=\s*polygon\s+\w+|\s*$)", re.DOTALL
)


class DSLPromptTemplate:
    """Defines prompt templates for RL training."""

    def __init__(
        self,
        context_fields: List[str],
        target_fields: List[str],
        context_template: str = "<context> {content} </context>",
        field_template: str = "<{field}> {value} </{field}>",
        separator: str = " ",
        probability_of_space_sampling: float = 0.5,
        min_space_sampling_ratio: float = 0.1,
        max_space_sampling_ratio: float = 0.6,
    ):
        """
        Initialize the prompt template with context fields and templates.

        Args:
            context_fields: List of fields to include in context
            target_fields: List of fields to include in target generation
            context_template: Template for wrapping context
            field_template: Template for individual fields
            separator: Separator between fields
            probability_of_space_sampling: Probability of sampling spaces (between 0 and 1)
            min_space_sampling_ratio: Minimum ratio of spaces to sample (between 0 and 1)
            max_space_sampling_ratio: Maximum ratio of spaces to sample (between 0 and 1)
        """
        self.context_fields = context_fields
        self.target_fields = target_fields
        self.context_template = context_template

        # Space sampling parameters
        if not 0.0 <= probability_of_space_sampling <= 1.0:
            raise ValueError("probability_of_space_sampling must be between 0 and 1")
        if not 0.0 <= min_space_sampling_ratio <= 1.0:
            raise ValueError("min_space_sampling_ratio must be between 0 and 1")
        if not 0.0 <= max_space_sampling_ratio <= 1.0:
            raise ValueError("max_space_sampling_ratio must be between 0 and 1")
        if min_space_sampling_ratio > max_space_sampling_ratio:
            raise ValueError(
                "min_space_sampling_ratio must be less than or equal to max_space_sampling_ratio"
            )

        self.probability_of_space_sampling = probability_of_space_sampling
        self.min_space_sampling_ratio = min_space_sampling_ratio
        self.max_space_sampling_ratio = max_space_sampling_ratio

        # Generate target template if not provided
        self.target_template = self._build_target_template(self.target_fields)
        self.field_template = field_template
        self.separator = separator

    def _build_target_template(self, target_fields: List[str]) -> str:
        """Build a <generate> template from selected target fields."""
        target_parts = [DSL_DATA_COLUMN_TO_DSL_TAG[field] for field in target_fields]
        target_content = " ".join(f"<{field}>" for field in target_parts)
        return f"<generate> {target_content} </generate>"

    def _extract_spaces(self, spaces_str: str) -> List[str]:
        """
        Extract individual space definitions from a spaces string.

        Each returned entry is a single polygon together with its holes block
        (if present), e.g.::

            "polygon core 0,0 100,0 100,100 0,100 holes { hole 10,10 20,10 20,20 10,20 }"

        Args:
            spaces_str: String containing space definitions

        Returns:
            List[str]: List of individual space definitions, each optionally
            including a ``holes { ... }`` block.
        """
        # Remove the outer 'spaces { }' wrapper
        spaces_content = spaces_str.strip()
        if spaces_content.startswith("spaces {") and spaces_content.endswith("}"):
            spaces_content = spaces_content[8:-1].strip()

        if not spaces_content:
            return []

        # Use a regex to split on polygon boundaries
        return [m.strip() for m in _RE_DSL_POLYGON.findall(spaces_content)]

    def _sample_spaces(self, spaces_str: str) -> Optional[str]:
        """
        Randomly sample a subset of spaces from the full spaces string.
        Excludes spaces with "undefined" labels.

        Args:
            spaces_str: Full spaces string

        Returns:
            Optional[str]: Sampled spaces string or None if no spaces are sampled
        """
        if "dsl_spaces" not in self.context_fields:
            return None

        # Extract individual spaces
        all_spaces = self._extract_spaces(spaces_str)

        if not all_spaces:
            return spaces_str

        # Filter out undefined spaces
        defined_spaces = [
            space for space in all_spaces if not space.startswith("polygon undefined")
        ]

        if not defined_spaces:
            return None

        # Sample a ratio between min and max space sampling ratios
        sample_ratio = random.uniform(
            self.min_space_sampling_ratio, self.max_space_sampling_ratio
        )
        num_to_sample = int(len(defined_spaces) * sample_ratio)

        # If no spaces are sampled, return None
        if num_to_sample == 0:
            return None

        # Sample spaces
        sampled_spaces = random.sample(defined_spaces, num_to_sample)

        # Reconstruct the spaces string
        return f"spaces {{ {' '.join(sampled_spaces)} }}"

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a single sample into a prompt.

        Args:
            sample: Dictionary containing the sample data with context fields

        Returns:
            str: Formatted prompt string
        """
        context_parts = []

        # Shuffle context fields to randomize order
        for field in random.sample(self.context_fields, len(self.context_fields)):
            if field in sample:
                value = sample[field]

                # Special handling for dsl_spaces
                if field == "dsl_spaces":
                    should_sample = random.random() < self.probability_of_space_sampling
                    if not should_sample:
                        continue

                    # Sample spaces based on the defined ratio
                    value = self._sample_spaces(value)
                    # If no spaces were sampled, skip this field
                    if value is None:
                        continue

                field_content = self.field_template.format(
                    field=DSL_DATA_COLUMN_TO_DSL_TAG[field], value=value
                )
                context_parts.append(field_content)

        # Combine context parts
        context_content = self.separator.join(context_parts)
        context = self.context_template.format(content=context_content)

        # Combine with target template
        target_template = self._build_target_template(self.target_fields)
        prompt = f"{context} {target_template}"
        return prompt


class DSLDataset(BaseDataset):
    """
    DSL-specific dataset.
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        context_fields: Optional[List[str]] = None,
        target_fields: Optional[List[str]] = None,
        context_template: str = "<context> {content} </context>",
        field_template: str = "<{field}> {value} </{field}>",
        prompt_separator: str = " ",
        probability_of_space_sampling: float = 0.5,
        min_space_sampling_ratio: float = 0.1,
        max_space_sampling_ratio: float = 0.6,
        **kwargs,
    ):
        """
        Initialize DSL TRL dataset.

        Args:
            dataset_name_or_path: HuggingFace dataset name or local path
            context_fields: List of DSL fields for context (default: common DSL fields)
            target_fields: List of DSL fields for target generation
            context_template: Template for wrapping context
            field_template: Template for individual fields
            prompt_separator: String to join fields
            probability_of_space_sampling: Probability of sampling spaces
            min_space_sampling_ratio: Min ratio of spaces to sample
            max_space_sampling_ratio: Max ratio of spaces to sample
            **kwargs: Additional arguments passed to BaseDataset (including train_nrows, eval_nrows, load_train)
        """
        super().__init__(dataset_name_or_path=dataset_name_or_path, **kwargs)

        # Default context fields if not provided
        if context_fields is None:
            context_fields = ["dsl_building", "dsl_structure", "dsl_mass", "dsl_spaces"]

        # Default target fields if not provided
        if target_fields is None:
            target_fields = ["dsl_spaces"]

        # Initialize prompt template
        self.prompt_template = DSLPromptTemplate(
            context_fields=context_fields,
            target_fields=target_fields,
            context_template=context_template,
            field_template=field_template,
            separator=prompt_separator,
            probability_of_space_sampling=probability_of_space_sampling,
            min_space_sampling_ratio=min_space_sampling_ratio,
            max_space_sampling_ratio=max_space_sampling_ratio,
        )

    def load_data(self) -> Dict[str, Dataset]:
        """
        Load and process DSL dataset.

        Returns:
            Dict mapping split names to processed Dataset objects
        """
        raw_datasets = self._load_and_subsample_dataset()
        datasets = {}

        # Determine which splits to process
        splits_to_process = []
        if self.train_split in raw_datasets:
            splits_to_process.append(self.train_split)
        if self.eval_split in raw_datasets:
            splits_to_process.append(self.eval_split)

        for split_name in splits_to_process:
            dataset = raw_datasets[split_name]

            # Add prompts
            dataset = self._add_prompts_to_dataset(dataset)

            # Store dataset
            if split_name == self.train_split:
                self.train_dataset = dataset
            else:
                self.eval_dataset = dataset

            datasets[split_name] = dataset
            logger.info("Loaded %s split: %d samples", split_name, len(dataset))

        return datasets

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a DSL sample into a prompt.

        Args:
            sample: Dictionary containing DSL fields

        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format_prompt(sample)
