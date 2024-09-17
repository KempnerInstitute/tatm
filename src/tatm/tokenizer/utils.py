import logging

from requests.exceptions import HTTPError
from tokenizers import Tokenizer

LOGGER = logging.getLogger(__name__)


def load_tokenizer(tokenizer_id: str) -> Tokenizer:
    """Load a tokenizer from a pretrained model or a file.

    Args:
        tokenizer_id: The identifier of the tokenizer, which can be a model name or a file path.

    Returns:
        Tokenizer: The loaded tokenizer.

    Raises:
        ValueError: If the tokenizer cannot be loaded.
    """
    try:
        tokenizer = Tokenizer.from_pretrained(tokenizer_id)
    except (HTTPError, ValueError):
        try:
            tokenizer = Tokenizer.from_file(tokenizer_id)
        except Exception as e:
            LOGGER.error(f"Failed to load tokenizer: {e}")
            raise ValueError(f"Invalid tokenizer: {tokenizer_id}") from e

    return tokenizer
