"""
FLOORA tokenizer with a vocabulary for DSL tokens.
"""

import os
from typing import List, Optional
import logging

from transformers import GPT2Tokenizer
from tokenizers import AddedToken

logger = logging.getLogger(__name__)

TOKENIZER_FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tokenizer_files"
)


class FLOORATokenizer(GPT2Tokenizer):
    """A GPT2 tokenizer with a vocabulary for FLOORA DSL tokens."""

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_prefix_space=True,
        add_bos_token=False,
        add_eos_token=True,
        **kwargs,
    ):
        """Initializes the FLOORA tokenizer.

        The DSL vocabulary is read from the `added_tokens.json`/
        `tokenizer_config.json` files bundled in `tokenizer_files/` whenever
        the tokenizer is loaded via `from_pretrained`.

        Args:
            vocab_file: Path to the vocabulary file.
            merges_file: Path to the merges file.
            errors: Paradigm to follow when decoding bytes to UTF-8.
            unk_token: The unknown token.
            bos_token: The beginning-of-sequence token.
            eos_token: The end-of-sequence token.
            pad_token: The padding token.
            add_prefix_space: Whether to add a leading space to the input.
            add_bos_token: Whether to prepend a bos token to sequences.
            add_eos_token: Whether to append an eos token to sequences.
            **kwargs: Additional keyword arguments forwarded to the parent
                tokenizer.
        """
        # Convert string tokens to AddedToken objects with proper lstrip settings
        if isinstance(bos_token, str):
            bos_token = AddedToken(
                bos_token, lstrip=True, rstrip=False, normalized=True
            )
        if isinstance(eos_token, str):
            eos_token = AddedToken(
                eos_token, lstrip=True, rstrip=False, normalized=True
            )
        if isinstance(pad_token, str):
            pad_token = AddedToken(
                pad_token, lstrip=True, rstrip=False, normalized=True
            )
        if isinstance(unk_token, str):
            unk_token = AddedToken(
                unk_token, lstrip=False, rstrip=False, normalized=True
            )

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        # Set start of completion (soc_token) and end of completion (eoc_token)
        self.soc_token = "<completion>"
        self.eoc_token = "</completion>"
        self.soc_token_id = self.convert_tokens_to_ids(self.soc_token)
        self.eoc_token_id = self.convert_tokens_to_ids(self.eoc_token)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path=None, *inputs, **kwargs
    ) -> "FLOORATokenizer":
        """Loads the FLOORA tokenizer, defaulting to the bundled DSL vocabulary.

        Args:
            pretrained_model_name_or_path: Repo id or directory to load from.
                Defaults to the `tokenizer_files/` bundled with this package.
            *inputs: Additional positional arguments forwarded to
                `PreTrainedTokenizerBase.from_pretrained`.
            **kwargs: Additional keyword arguments forwarded to
                `PreTrainedTokenizerBase.from_pretrained`.

        Returns:
            The loaded `FLOORATokenizer` instance.
        """
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = TOKENIZER_FILES_DIR
        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Builds model inputs by adding bos/eos tokens around given sequences.

        Overrides the base GPT2 tokenizer, which lacks eos token support, to
        automatically add bos/eos tokens when `add_bos_token`/`add_eos_token`
        are enabled.

        Args:
            token_ids_0: List of IDs for the first sequence.
            token_ids_1: Optional list of IDs for a second sequence.

        Returns:
            List of input IDs with the appropriate special tokens added.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Retrieves a mask indicating which tokens are special tokens.

        Called when adding special tokens via the tokenizer's
        `prepare_for_model` method, for a token list that has no special
        tokens added yet.

        Args:
            token_ids_0: List of IDs.
            token_ids_1: Optional second list of IDs for sequence pairs.
            already_has_special_tokens: Whether the token list is already
                formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0
            for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def __call__(self, text, add_special_tokens=True, **kwargs):
        """Encodes text, always adding special tokens.

        Special tokens are always added regardless of the parameter because
        all tokens are effectively special tokens in the DSL. Callers may
        pass `add_special_tokens=False` expecting it to be honored.

        Args:
            text: The text or list of texts to encode.
            add_special_tokens: Whether to add special tokens (ignored).
            **kwargs: Additional keyword arguments passed to `__call__`.

        Returns:
            The tokenizer's `BatchEncoding` output.
        """
        return super().__call__(text, add_special_tokens=True, **kwargs)

    def encode(self, text, add_special_tokens=True, **kwargs):
        """Encodes text, always adding special tokens.

        Special tokens are always added regardless of the parameter because
        all tokens are effectively special tokens in the DSL. Callers may
        pass `add_special_tokens=False` expecting it to be honored.

        Args:
            text: The text or list of texts to encode.
            add_special_tokens: Whether to add special tokens (ignored).
            **kwargs: Additional keyword arguments passed to `encode`.

        Returns:
            List of encoded token IDs.
        """
        return super().encode(text, add_special_tokens=True, **kwargs)

    def decode(self, sequences, skip_special_tokens=False, **kwargs):
        """Decodes token IDs, always keeping special tokens.

        Special tokens are always kept regardless of the parameter because
        all tokens are effectively special tokens in the DSL. Callers may
        pass `skip_special_tokens=True` expecting it to be honored.

        Args:
            sequences: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens (ignored).
            **kwargs: Additional keyword arguments passed to `decode`.

        Returns:
            The decoded string.
        """
        return super().decode(sequences, skip_special_tokens=False, **kwargs)

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        """Decodes batches of token IDs, always keeping special tokens.

        Special tokens are always kept regardless of the parameter because
        all tokens are effectively special tokens in the DSL. Callers may
        pass `skip_special_tokens=True` expecting it to be honored.

        Args:
            sequences: List of token ID lists to decode.
            skip_special_tokens: Whether to skip special tokens (ignored).
            **kwargs: Additional keyword arguments passed to `batch_decode`.

        Returns:
            List of decoded strings.
        """
        return super().batch_decode(sequences, skip_special_tokens=False, **kwargs)

    def convert_tokens_to_string(self, tokens):
        """Converts a list of tokens to a string, handling None tokens.

        None tokens can occur when the model generates token IDs outside the
        vocabulary.

        Args:
            tokens: List of token strings, which may contain None values.

        Returns:
            The decoded string with any None tokens replaced by `unk_token`.
        """
        # Single pass: filter tokens and collect None indices
        none_indices = []
        filtered_tokens = []
        for i, token in enumerate(tokens):
            if token is None:
                none_indices.append(i)
                filtered_tokens.append(self.unk_token)
            else:
                filtered_tokens.append(token)

        # Log warning if None tokens were found
        if none_indices:
            logger.warning(
                "Found %d None token(s) at position(s) %s during decoding. "
                "This indicates the model generated token IDs outside the vocabulary. "
                "Replacing with '%s' token.",
                len(none_indices),
                (none_indices[:10] if len(none_indices) > 10 else none_indices),
                self.unk_token,
            )

        return super().convert_tokens_to_string(filtered_tokens)
