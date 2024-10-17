"""Module holding classes for handling curated data available
within the Tatm Data Garden.

A note on terminology:
We use "Data" to refer to curated data available w/in the Kempner testbed collection
while we use "Dataset" to refer to structures used to present data in a format that
can be used by external modeling systems (e.g. PyTorch, JAX, etc.). This distinction
is entirely arbitrary but is intended to help separate functionality and to
match function with pytorch naming conventions.
"""

import pathlib
from abc import ABC, abstractmethod
from typing import Union

import datasets

from tatm.data.metadata import DataContentType, TatmDataMetadata

datasets.disable_caching()


class TatmData(ABC):
    """Generic dataset class, provides interface to access multiple types of datasets.

    Args:
        metadata: Metadata object.
    """

    def __init__(self, metadata: TatmDataMetadata, *args, **kwargs):
        self.metadata = metadata
        self.initialize(*args, **kwargs)

    @classmethod
    @abstractmethod
    def from_metadata(cls, metadata: TatmDataMetadata) -> "TatmData":
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


class TatmTextData(TatmData):
    """Text dataset class, provides interface to access text datasets.

    Args:
        metadata: Metadata object.
    """

    @classmethod
    def from_metadata(
        cls, metadata: TatmDataMetadata, corpus=None, split="train"
    ) -> "TatmTextData":
        """Create a TatumTextDataset object from metadata.

        Args:
            metadata: Metadata object.

        Returns:
            TatumTextDataset: Text dataset object.
        """
        if metadata.data_content != DataContentType.TEXT:
            raise ValueError("Metadata does not describe a text dataset.")
        return cls(metadata, corpus=corpus, split=split)

    def initialize(self, corpus: str = None, split: str = "train"):
        """Initialize the dataset.

        Args:
            corpus: Corpus to load. Defaults to None.
        """
        if (
            self.metadata.corpus_separation_strategy == "configs"
            or self.metadata.corpus_separation_strategy is None  # noqa: W503
        ):
            self.dataset = datasets.load_dataset(
                self.metadata.dataset_path,
                name=corpus,
                streaming=True,
                trust_remote_code=True,
            )[split]
        elif self.metadata.corpus_separation_strategy == "data_dirs":
            if self.metadata.corpus_data_dir_parent is not None:
                corpus_path = (
                    pathlib.Path(self.metadata.corpus_data_dir_parent) / corpus
                )
            else:
                corpus_path = corpus
            self.dataset = datasets.load_dataset(
                self.metadata.dataset_path,
                data_dir=corpus_path,
                streaming=True,
                trust_remote_code=True,
            )[split]

    def __iter__(self):
        """Iterate over the dataset."""
        self.data_iter = iter(self.dataset)
        return self

    def __next__(self):
        """Get the next item from the dataset."""
        return next(self.data_iter)


def get_data(identifier: Union[str, TatmDataMetadata]) -> TatmData:
    """Get a dataset object from an identifier.

    Args:
        identifier : Identifier for the dataset.

    Returns:
        TatmDataset: Dataset object.
    """
    if isinstance(identifier, str):
        split_identifier = identifier.split(":")
        identifier = split_identifier[0]
        if len(split_identifier) > 1:
            corpus = split_identifier[1]
        else:
            corpus = None
        return _dataset_from_path(identifier, corpus=corpus)
    elif isinstance(identifier, TatmDataMetadata):
        return _dataset_from_metadata(identifier, corpus=corpus)


def _dataset_from_path(path: str, corpus=None) -> TatmData:
    """Create a dataset object from a path.

    Args:
        path: Path to the dataset.

    Returns:
        TatmDataset: Dataset object.
    """
    path = pathlib.Path(path)
    if (path / "metadata.yaml").exists():
        metadata = TatmDataMetadata.from_yaml(path / "metadata.yaml")
    elif (path / "metadata.json").exists():
        metadata = TatmDataMetadata.from_json(path / "metadata.json")
    else:
        raise ValueError(
            (
                "No metadata file found in dataset path. "
                "The metadata file must be named 'metadata.yaml' or 'metadata.json'."
            )
        )
    return _dataset_from_metadata(metadata, corpus=corpus)


def _dataset_from_metadata(metadata: TatmDataMetadata, corpus=None) -> TatmData:
    """Create a dataset object from metadata.

    Args:
        metadata: Metadata object.

    Returns:
        TatmDataset: Dataset object.
    """
    if metadata.data_content == DataContentType.TEXT:
        return TatmTextData.from_metadata(metadata, corpus=corpus)
    else:
        raise NotImplementedError(
            f"Data content type {metadata.data_content} is not yet supported."
        )
