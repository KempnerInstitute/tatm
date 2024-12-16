import logging

from tatm.config import load_config
from tatm.data.metadata_store.metadata_backend import (
    JsonTatmMetadataStoreBackend,
    TatmMetadataStoreBackend,
)

BACKEND: TatmMetadataStoreBackend = None
BACKEND_INITIALIZED = False

LOGGER = logging.getLogger(__name__)


def get_metadata(name: str) -> str:
    """Get metadata for a dataset by name.

    Args:
        name: Name of the dataset to lookup.

    Returns:
        str: JSON representation of the metadata.
    """
    if not BACKEND_INITIALIZED:
        set_backend()
    if BACKEND is None:
        LOGGER.warning(
            "Call to metadata store get_metadata() with no backend set. Metadata store will not be used."
        )
        return None
    try:
        return BACKEND.lookup(name)
    except KeyError:
        LOGGER.warning(f"Metadata not found for dataset: {name}")
        return None


def set_backend() -> None:
    """Set the metadata store backend.

    Args:
        backend: Metadata store backend.
    """
    global BACKEND
    global BACKEND_INITIALIZED

    BACKEND_INITIALIZED = True

    cnf = load_config()
    backend_type = cnf.metadata_backend.type
    print(backend_type)
    if backend_type == "json":
        if "metadata_store_path" not in cnf.metadata_backend.args:
            cnf.metadata_backend.args["metadata_store_path"] = "metadata.json"
        BACKEND = JsonTatmMetadataStoreBackend(**cnf.metadata_backend.args)
    elif backend_type is None:
        LOGGER.warning(
            "No metadata store backend set. Metadata store will not be used."
        )
        return
    else:
        raise ValueError(
            f"Unknown metadata store backend type specified in TATM_METADATA_STORE_PATH: {backend_type}"
        )


def reset_backend() -> None:
    """Reset the metadata store backend."""
    global BACKEND
    global BACKEND_INITIALIZED

    BACKEND = None
    BACKEND_INITIALIZED = False
