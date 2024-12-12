import json

import numpy as np
import pytest
import yaml

from tatm.data import get_data, get_dataset
from tatm.data.metadata import DataContentType, TatmDataMetadata
from tatm.data.metadata_store import get_metadata
from tatm.data.metadata_store.interface import reset_backend

from .test_memmap_dataset import sample_dataset  # noqa: F401


@pytest.fixture
def json_metadata_store(tmp_path, sample_dataset):  # noqa: F811
    """Pytest Fixture creating a metadata store with JSON metadata and a config file pointing to it."""
    metadata_store = tmp_path / "metadata.json"
    config_path = tmp_path / "config.json"
    config = {
        "metadata_backend": {
            "type": "json",
            "args": {"metadata_store_path": str(metadata_store)},
        }
    }
    with open(config_path, "w") as f:
        f.write(yaml.dump(config))

    metadata = TatmDataMetadata(
        name="dataset1",
        dataset_path="./",
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DataContentType.TEXT,
    )
    other_metadata = TatmDataMetadata.from_directory("tests/data/json_data")
    tokenized_metadata = TatmDataMetadata.from_directory(sample_dataset[0])

    out = {
        "dataset1": metadata.as_json(),
        "dataset2": other_metadata.as_json(),
        "tokenized": tokenized_metadata.as_json(),
    }
    with open(metadata_store, "w") as f:
        json.dump(out, f)
    yield metadata_store, config_path


def test_get_metadata(json_metadata_store, monkeypatch):
    reset_backend()
    _, config_path = json_metadata_store
    monkeypatch.setenv("TATM_BASE_CONFIG", str(config_path))
    metadata = TatmDataMetadata(
        name="dataset1",
        dataset_path="./",
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DataContentType.TEXT,
    )

    assert get_metadata("dataset1") == metadata.as_json()


def test_get_data_from_metadata_store(json_metadata_store, monkeypatch):
    reset_backend()
    _, config_path = json_metadata_store
    monkeypatch.setenv("TATM_BASE_CONFIG", str(config_path))
    data = get_data("dataset2")
    first_item = next(iter(data))
    assert first_item["text"] == "hello world"


def test_get_tokenized_from_metadata_store(json_metadata_store, monkeypatch):
    reset_backend()
    _, config_path = json_metadata_store
    monkeypatch.setenv("TATM_BASE_CONFIG", str(config_path))
    dataset = get_dataset("tokenized", context_length=100)
    assert len(dataset) == 100
    assert dataset.vocab_size == 32100
    assert np.all(dataset[0]["token_ids"] == np.arange(100))
    assert np.all(dataset[20]["token_ids"] == np.arange(100) + 2000)
    assert isinstance(dataset[0]["token_ids"], np.ndarray)
    assert not isinstance(dataset[0]["token_ids"], np.memmap)
    assert isinstance(dataset[0]["document_ids"], np.ndarray)
    assert not isinstance(dataset[0]["document_ids"], np.memmap)
    assert dataset.num_files() == 10
    assert dataset.num_tokens() == 100 * 100
