import logging
from pathlib import Path

from tokenizers import __version__ as tk_version

from tatm.data.metadata import DataContentType, TatmDataMetadata, TokenizedDataMetadata
from tatm.tokenizer.utils import load_tokenizer
from tatm.version import __version__ as tatm_version

LOGGER = logging.getLogger(__name__)


def write_metadata(
    tokenizer: str,
    file_prefix: str,
    output_dir: str,
    dtype: str = "uint16",
    file_extension: str = "bin",
    data_description=None,
) -> None:
    """_summary_

    Args:
        tokenizer: name of a huggingface tokenizer or path to a tokenizer file
        file_prefix: the file prefix for the tokenized files in the output directory
        output_dir (str): the directory where the tokenized files and metadata will be saved
        dtype: the datatype of the token IDs on disk. Defaults to "uint16".
        file_extension: The file extension of the token array files. Defaults to "bin".
        data_description: Dataset description to be included in the dataset metadata. Defaults to None.
    """
    tokenizer_id = tokenizer
    tokenizer = load_tokenizer(tokenizer_id)

    tokenizer.save(str(Path(output_dir) / "tokenizer.json"))
    tokenizer_metadata = TokenizedDataMetadata(
        tokenizer=tokenizer_id,
        file_prefix=str(Path(output_dir).resolve() / file_prefix),
        dtype=dtype,
        file_extension=file_extension,
        vocab_size=tokenizer.get_vocab_size(),
        tatm_version=tatm_version,
        tokenizers_version=tk_version,
    )
    if data_description is None:
        data_description = "Tokenized dataset created using tatm"
    metadata = TatmDataMetadata(
        name=file_prefix,
        dataset_path=output_dir,
        description=data_description,
        date_downloaded="",
        download_source="Tokenized by tatm",
        data_content=DataContentType.TEXT,
        content_field="NA",
        corpuses=[],
        tokenized_info=tokenizer_metadata,
    )
    metadata.to_yaml(Path(output_dir) / "metadata.yaml")  # Save metadata to JSON
