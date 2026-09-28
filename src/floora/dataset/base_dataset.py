"""
Base data class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """
    Abstract base dataset class.
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        train_split: str = "train",
        eval_split: str = "test",
        dataset_config: Optional[str] = None,
        num_workers: int = 4,
        train_nrows: Optional[Union[int, float]] = None,
        eval_nrows: Optional[Union[int, float]] = None,
    ):
        """
        Initialize base dataset.

        Args:
            dataset_name_or_path: HuggingFace dataset name or local path
            train_split: Name of training split
            eval_split: Name of evaluation split
            dataset_config: Optional dataset configuration/subset name
            num_workers: Number of workers for data loading
            train_nrows: Number of training rows to use. If float (0-1), treats as fraction.
                        If int, selects that many random samples.
            eval_nrows: Number of evaluation rows to use. If float (0-1), treats as fraction.
                       If int, selects that many random samples.
        """
        self.dataset_name_or_path = dataset_name_or_path
        self.train_split = train_split
        self.eval_split = eval_split
        self.dataset_config = dataset_config
        self.num_workers = num_workers
        self.train_nrows = train_nrows
        self.eval_nrows = eval_nrows

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    @abstractmethod
    def load_data(self) -> Dict[str, Dataset]:
        """
        Load and process the dataset.

        Returns:
            Dict mapping split names to Dataset objects
        """
        raise NotImplementedError

    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a single sample into a prompt string.

        Args:
            sample: Dictionary containing sample data

        Returns:
            Formatted prompt string
        """
        raise NotImplementedError

    def _subsample_dataset(
        self, dataset: Dataset, nrows: Optional[Union[int, float]], split_name: str
    ) -> Dataset:
        """
        Subsample dataset based on nrows parameter.

        Args:
            dataset: Dataset to subsample
            nrows: Number of rows to keep. If float (0-1), treats as fraction.
                  If int, selects that many random samples.
            split_name: Name of split for logging

        Returns:
            Subsampled dataset
        """
        if nrows is None:
            return dataset

        total_rows = len(dataset)

        if isinstance(nrows, float):
            if not 0 < nrows <= 1:
                raise ValueError(
                    f"When nrows is a float, it must be between 0 and 1, got {nrows}"
                )
            num_samples = int(total_rows * nrows)
            logger.info(
                "Subsampling %s split: %d rows (%.1f%% of %d total)",
                split_name,
                num_samples,
                nrows * 100,
                total_rows,
            )
        elif isinstance(nrows, int):
            if nrows <= 0:
                raise ValueError(
                    f"When nrows is an int, it must be positive, got {nrows}"
                )
            num_samples = min(nrows, total_rows)
            logger.info(
                "Subsampling %s split: %d rows (out of %d total)",
                split_name,
                num_samples,
                total_rows,
            )
        else:
            raise TypeError(f"nrows must be int or float, got {type(nrows).__name__}")

        # Shuffle and select random samples
        return dataset.shuffle(seed=42).select(range(num_samples))

    def _load_raw_dataset(self) -> Dict[str, Dataset]:
        """
        Load raw dataset from HuggingFace or a local path.

        Returns:
            Dict mapping split names to raw Dataset objects
        """
        logger.info("Loading dataset from: %s", self.dataset_name_or_path)
        if self.dataset_config:
            logger.info("Using dataset config/subset: %s", self.dataset_config)
            dataset = load_dataset(
                self.dataset_name_or_path,
                self.dataset_config,
            )
        else:
            dataset = load_dataset(
                self.dataset_name_or_path,
            )
        return dataset

    def _load_and_subsample_dataset(self) -> Dict[str, Dataset]:
        """
        Load raw dataset and apply subsampling.

        This method should be used by subclasses instead of _load_raw_dataset()
        to ensure consistent subsampling behavior.

        Returns:
            Dict mapping split names to subsampled Dataset objects
        """
        raw_datasets = self._load_raw_dataset()

        # Only subsample splits that were actually loaded
        splits_to_process = []
        if self.train_split in raw_datasets:
            splits_to_process.append((self.train_split, self.train_nrows))
        if self.eval_split in raw_datasets:
            splits_to_process.append((self.eval_split, self.eval_nrows))

        for split_name, nrows in splits_to_process:
            if nrows is not None:
                raw_datasets[split_name] = self._subsample_dataset(
                    raw_datasets[split_name], nrows, split_name
                )

        return raw_datasets

    def _add_prompts_to_dataset(self, dataset: Dataset) -> Dataset:
        """
        Add formatted prompts to dataset.

        Args:
            dataset: Dataset to process

        Returns:
            Dataset with 'prompt' field added
        """

        def add_prompt(sample):
            sample["prompt"] = self.format_prompt(sample)
            return sample

        return dataset.map(
            add_prompt, load_from_cache_file=False, num_proc=self.num_workers
        )
