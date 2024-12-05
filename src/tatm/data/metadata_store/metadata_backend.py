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

        Returns:
            str: JSON representation of the metadata.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


class JsonTatmMetadataStoreBackend(TatmMetadataStoreBackend):
    """Metadata store backend that stores metadata as JSON files."""

    def __init__(self, metadata_store_path: str, *args, **kwargs):
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
            current_level = metadata[name]
            while True:
                # This is a hack to get around the fact that the metadata can be stored as a string and there may be string escapes in the metadata
                next_level = json.loads(current_level)
                if not isinstance(next_level, str):
                    return current_level
                current_level = next_level
        return json.dumps(metadata[name])
