import pathlib
from typing import Generator

import tatm.data


def metadata_files(
    directory, include_tokenized=False
) -> Generator[tuple[str, pathlib.Path, tatm.data.TatmDataMetadata], None, None]:
    root = pathlib.Path(directory)
    for metadata_file in root.rglob("metadata.*"):
        if include_tokenized or "tokenized" not in metadata_file.parts:
            data_genre = metadata_file.parts[2]
            metadata = tatm.data.TatmDataMetadata.from_file(metadata_file)
            yield data_genre, metadata_file.parent, metadata


def tokenized_datasets(
    parent_path: pathlib.Path | str,
) -> Generator[tuple[pathlib.Path, tatm.data.TatmDataMetadata], None, None]:
    if not isinstance(parent_path, pathlib.Path):
        parent_path = pathlib.Path(parent_path)
    tokenized_data_path = parent_path / "tokenized"
    if not tokenized_data_path.exists():
        return
    for tokenized_dataset in tokenized_data_path.rglob("metadata.*"):
        metadata = tatm.data.TatmDataMetadata.from_file(tokenized_dataset)
        if metadata.tokenized_info:
            yield tokenized_dataset.parent, metadata
