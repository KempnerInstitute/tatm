import json

import numpy as np
import pytest

from tatm.data import get_data, get_dataset
from tatm.data.metadata import DataContentType, TatmDataMetadata
from tatm.data.metadata_store import get_metadata
from tatm.data.metadata_store.interface import reset_backend

from .test_memmap_dataset import sample_dataset  # noqa: F401


@pytest.fixture
def json_metadata_store(tmp_path, sample_dataset):  # noqa: F811
    metadata_store = tmp_path / "metadata.json"
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
    yield metadata_store


def test_get_metadata(json_metadata_store, monkeypatch):
    reset_backend()
    monkeypatch.setenv("TATM_METADATA_STORE_BACKEND", "json")
    monkeypatch.setenv("TATM_METADATA_STORE_PATH", str(json_metadata_store))
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
    monkeypatch.setenv("TATM_METADATA_STORE_BACKEND", "json")
    monkeypatch.setenv("TATM_METADATA_STORE_PATH", str(json_metadata_store))
    data = get_data("dataset2")
    first_item = next(iter(data))
    assert first_item["text"] == "hello world"


def test_get_tokenized_from_metadata_store(json_metadata_store, monkeypatch):
    reset_backend()
    monkeypatch.setenv("TATM_METADATA_STORE_BACKEND", "json")
    monkeypatch.setenv("TATM_METADATA_STORE_PATH", str(json_metadata_store))
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
