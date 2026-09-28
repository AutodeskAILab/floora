"""
Utility functions for registering custom FLOORA tokenizers with HuggingFace transformers.
"""

import logging
import transformers
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from floora.tokenizer import FLOORATokenizer

logger = logging.getLogger(__name__)


def register_custom_tokenizers():
    """
    Register the FLOORA tokenizer with HuggingFace transformers.
    Call this before initializing vLLM with the custom tokenizer.
    """
    is_fast = issubclass(FLOORATokenizer, PreTrainedTokenizerFast)

    if is_fast:
        AutoTokenizer.register(
            config_class=None,
            slow_tokenizer_class=None,
            fast_tokenizer_class=FLOORATokenizer,
        )
    else:
        AutoTokenizer.register(
            config_class=None,
            slow_tokenizer_class=FLOORATokenizer,
            fast_tokenizer_class=None,
        )

    # Make available in transformers namespace
    setattr(transformers, FLOORATokenizer.__name__, FLOORATokenizer)
    logger.info("Registered custom tokenizer: %s", FLOORATokenizer.__name__)
