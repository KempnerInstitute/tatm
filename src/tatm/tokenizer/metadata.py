import logging
from importlib.metadata import version
from pathlib import Path
<<<<<<< HEAD
from typing import List
=======
>>>>>>> main

from tatm.data.metadata import (
    DataContentType,
    TatmDataMetadata,
    TokenizedMetadataComponenet,
)
from tatm.tokenizer.utils import load_tokenizer

LOGGER = logging.getLogger(__name__)


def write_metadata(
    tokenizer: str,
    output_dir: str,
    file_prefix: str,
    dtype: str = "uint16",
    file_extension: str = "bin",
<<<<<<< HEAD
    parent_datasets: List[str] = None,
=======
>>>>>>> main
    data_description=None,
) -> None:
    """_summary_

    Args:
        tokenizer: name of a huggingface tokenizer or path to a tokenizer file
        output_dir: the directory where the tokenized files and metadata will be saved
        file_prefix: the file prefix for the tokenized files in the output directory
        dtype: the datatype of the token IDs on disk. Defaults to "uint16".
        file_extension: The file extension of the token array files. Defaults to "bin".
        data_description: Dataset description to be included in the dataset metadata. Defaults to None.
    """
    tokenizer_id = tokenizer
    tokenizer = load_tokenizer(tokenizer_id)

    tokenizer.save(str(Path(output_dir) / "tokenizer.json"))
    tokenizer_metadata = TokenizedMetadataComponenet(
        tokenizer=tokenizer_id,
        file_prefix=str(Path(output_dir).resolve() / file_prefix),
        dtype=dtype,
        file_extension=file_extension,
        vocab_size=tokenizer.get_vocab_size(),
        tatm_version=version("tatm"),
        tokenizers_version=version("tokenizers"),
    )
<<<<<<< HEAD
    if parent_datasets is not None:
        tokenizer_metadata.parent_datasets = parent_datasets

=======
>>>>>>> main
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
