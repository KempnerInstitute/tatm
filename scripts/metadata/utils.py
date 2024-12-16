import pathlib
from typing import Generator, List

import tatm.data


def metadata_files(
    directory: pathlib.Path, include_tokenized=False, exclude_dirs: List[str] = None
) -> Generator[tuple[str, pathlib.Path, tatm.data.TatmDataMetadata], None, None]:
    """Recursively walk a directory and yield all tatm data metadata files. Can limit to non-tokenized metadata files and ignore certain directories,
    to avoid duplicates or large numbers of files that can slow down processing.

    Args:
        directory: The root directory to start the search from
        include_tokenized: Whether or not to return tokenized data sets fron the generator. Tokenized datasets are assumed to
          be stored in a sub-directory named "tokenized", and therefore if this is False, all metadata objects in such a 
          directory will be ignored. Defaults to False.
        exclude_dirs: A list of directory names to be ignored when searching. Any directory at any level matching this name
         will not have its contents included. Defaults to None.

    Yields:
        tuple[str, pathlib.Path, tatm.data.TatmDataMetadata]: A tuple containing the first sub-directory below the root (assumed to be the 
        general context of the data), the path to the directory holding metadata file, and the metadata object.
    """
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
    """Iterate over all tokenized datasets in a directory. Finds all metadata files in the "tokenized" sub-directory of the parent path.

    Args:
        parent_path: The parent directory to search for tokenized datasets. Assumed to be a non-tokenized dataset directory.

    Yields:
        tuple[pathlib.Path, tatm.data.TatmDataMetadata]: A tuple containing the path to the directory holding the tokenized dataset and the metadata object.
    """
    if not isinstance(parent_path, pathlib.Path):
        parent_path = pathlib.Path(parent_path)
    tokenized_data_path = parent_path / "tokenized"
    if not tokenized_data_path.exists():
        return
    for tokenized_dataset in tokenized_data_path.rglob("metadata.*"):
        metadata = tatm.data.TatmDataMetadata.from_file(tokenized_dataset)
        if metadata.tokenized_info:
            yield tokenized_dataset.parent, metadata
