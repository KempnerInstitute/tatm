import pathlib
from typing import Generator, List

import tatm.data


def metadata_files(
    directory: pathlib.Path, include_tokenized=False, exclude_dirs: List[str]=None
) -> Generator[tuple[str, pathlib.Path, tatm.data.TatmDataMetadata], None, None]:
    root = pathlib.Path(directory)
    for dir, dirs, _ in root.walk():
        if exclude_dirs is not None:
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

        metadata_files = dir.glob("metadata.*")
        for metadata_file in metadata_files:
            if include_tokenized or "tokenized" not in metadata_file.parts:
                data_genre = metadata_file.relative_to(root).parts[0]
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
