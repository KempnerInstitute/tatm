import pathlib
from abc import ABC, abstractmethod
from typing import Union

import datasets

from tatm.data.metadata import DatasetContentType, DatasetMetadata

datasets.disable_caching()


class TatmDataset(ABC):
    """Generic dataset class, provides interface to access multiple types of datasets.

    Args:
        metadata: Metadata object.
    """

    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        self.initialize()

    @classmethod
    @abstractmethod
    def from_metadata(cls, metadata: DatasetMetadata) -> "TatmDataset":
        """Create a dataset object from metadata.

        Args:
            metadata: Metadata object.

        Returns:
            TatmDataset: Dataset object.
        """

        raise NotImplementedError("This method must be implemented in a subclass.")

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """Initialize the dataset."""
        raise NotImplementedError("This method must be implemented in a subclass.")


class TatmTextDataset(TatmDataset):
    """Text dataset class, provides interface to access text datasets.

    Args:
        metadata: Metadata object.
    """

    @classmethod
    def from_metadata(cls, metadata: DatasetMetadata) -> "TatmTextDataset":
        """Create a TatumTextDataset object from metadata.

        Args:
            metadata: Metadata object.

        Returns:
            TatumTextDataset: Text dataset object.
        """
        if metadata.data_content != DatasetContentType.TEXT:
            raise ValueError("Metadata does not describe a text dataset.")
        return cls(metadata)

    def initialize(self, corpus: str = None, split: str = "train"):
        """Initialize the dataset.

        Args:
            corpus: Corpus to load. Defaults to None.
        """
        self.dataset = datasets.load_dataset(
            self.metadata.dataset_path, streaming=True
        )[split]

    def __iter__(self):
        """Iterate over the dataset."""
        self.data_iter = iter(self.dataset)
        return self

    def __next__(self):
        """Get the next item from the dataset."""
        return next(self.data_iter)


def get_dataset(identifier: Union[str, DatasetMetadata]) -> TatmDataset:
    """Get a dataset object from an identifier.

    Args:
        identifier : Identifier for the dataset.

    Returns:
        TatmDataset: Dataset object.
    """
    if isinstance(identifier, str):
        return _dataset_from_path(identifier)
    elif isinstance(identifier, DatasetMetadata):
        return _dataset_from_metadata(identifier)


def _dataset_from_path(path: str) -> TatmDataset:
    """Create a dataset object from a path.

    Args:
        path: Path to the dataset.

    Returns:
        TatmDataset: Dataset object.
    """
    path = pathlib.Path(path)
    if (path / "metadata.yaml").exists():
        metadata = DatasetMetadata.from_yaml(path / "metadata.yaml")
    elif (path / "metadata.json").exists():
        metadata = DatasetMetadata.from_json(path / "metadata.json")
    else:
        raise ValueError(
            (
                "No metadata file found in dataset path. "
                "The metadata file must be named 'metadata.yaml' or 'metadata.json'."
            )
        )
    metadata.dataset_path = str(path)
    return _dataset_from_metadata(metadata)


def _dataset_from_metadata(metadata: DatasetMetadata) -> TatmDataset:
    """Create a dataset object from metadata.

    Args:
        metadata: Metadata object.

    Returns:
        TatmDataset: Dataset object.
    """
    if metadata.data_content == DatasetContentType.TEXT:
        return TatmTextDataset.from_metadata(metadata)
    else:
        raise NotImplementedError(
            f"Data content type {metadata.data_content} is not yet supported."
        )
