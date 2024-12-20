import json
from abc import ABC, abstractmethod


class TatmMetadataStoreBackend(ABC):
    """Abstract class for metadata store backend.
    Responsible for storing and retrieving metadata.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def lookup(self, name: str) -> str:
        """Lookup metadata for a dataset by name.
        Should return a string containing a json representation of the metadata.

        Args:
            name: Name of the dataset to lookup.

        Raises:
            KeyError: If metadata is not found for the dataset.

        Returns:
            str: JSON representation of the metadata.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


class JsonTatmMetadataStoreBackend(TatmMetadataStoreBackend):
    def __init__(self, metadata_store_path: str, *args, **kwargs):
        """Metadata store backend that stores metadata as JSON files.

        Args:
            metadata_store_path: Path to the JSON serving as a metadata store.
        """
        super().__init__(*args, **kwargs)
        self.metadata_store_path = metadata_store_path

    def lookup(self, name: str) -> str:
        """Lookup metadata for a dataset by name.
        Should return a string containing a json representation of the metadata.

        Args:
            name: Name of the dataset to lookup.

        Returns:
            str: JSON representation of the metadata.
        """
        with open(self.metadata_store_path) as f:
            metadata = json.load(f)
        if isinstance(metadata[name], str):
            return metadata[name]

        return json.dumps(metadata[name])
