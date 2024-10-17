import json
import pathlib

import pytest
import yaml

from tatm.data.metadata import (
    DataContentType,
    TatmDataMetadata,
    create_metadata_interactive,
)


@pytest.fixture
def json_metadata(tmp_path):
    metadata = TatmDataMetadata(
        name="test",
        dataset_path=str(tmp_path),
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DataContentType.TEXT,
        content_field="text",
    )
    metadata.to_json(tmp_path / "metadata.json")
    yield (tmp_path, "metadata.json")
    return tmp_path / "metadata.json"


@pytest.fixture
def yaml_metadata(tmp_path):
    metadata = TatmDataMetadata(
        name="test",
        dataset_path=str(tmp_path),
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DataContentType.TEXT,
        content_field="text",
    )
    metadata.to_yaml(tmp_path / "metadata.yaml")
    yield (tmp_path, "metadata.yaml")
    return tmp_path / "metadata.yaml"


@pytest.fixture
def yml_metadata(tmp_path):
    metadata = TatmDataMetadata(
        name="test",
        dataset_path=str(tmp_path),
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DataContentType.TEXT,
        content_field="text",
    )
    metadata.to_yaml(tmp_path / "metadata.yml")
    yield (tmp_path, "metadata.yml")
    return tmp_path / "metadata.yml"


def test_json_load():
    filename = "tests/data/metadata_test.json"
    metadata = TatmDataMetadata.from_json(filename)
    test_file_dir = pathlib.Path(filename).resolve().parent
    assert metadata.name == "test"
    assert metadata.dataset_path == str(test_file_dir)
    assert metadata.description == "A test metadata file."
    assert metadata.date_downloaded == "2021-01-01"
    assert metadata.download_source == "http://example.com"
    assert metadata.data_content == DataContentType.TEXT
    assert metadata.content_field == "text"


def test_yaml_load():
    filename = "tests/data/metadata_test.yaml"
    metadata = TatmDataMetadata.from_yaml(filename)
    test_file_dir = pathlib.Path(filename).resolve().parent
    assert metadata.name == "test"
    assert metadata.dataset_path == str(test_file_dir)
    assert metadata.description == "A test metadata file."
    assert metadata.date_downloaded == "2021-01-01"
    assert metadata.download_source == "http://example.com"
    assert metadata.data_content == DataContentType.TEXT
    assert metadata.content_field == "text"


def test_json_save(tmp_path):
    metadata = TatmDataMetadata(
        name="test",
        dataset_path="./",
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DataContentType.TEXT,
    )
    metadata.to_json(tmp_path / "metadata_test.json")
    json_dict = json.load(open(tmp_path / "metadata_test.json"))
    assert json_dict["name"] == "test"
    assert json_dict["dataset_path"] == "./"
    assert json_dict["description"] == "A test metadata file."
    assert json_dict["date_downloaded"] == "2021-01-01"
    assert json_dict["download_source"] == "http://example.com"
    assert json_dict["data_content"] == "text"
    assert json_dict["content_field"] == "text"


def test_yaml_save(tmp_path):
    metadata = TatmDataMetadata(
        name="test",
        dataset_path="./",
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DataContentType.TEXT,
    )
    metadata.to_yaml(tmp_path / "metadata_test.yaml")
    with open(tmp_path / "metadata_test.yaml", "r") as f:
        yaml_dict = yaml.safe_load(f)
    assert yaml_dict["name"] == "test"
    assert yaml_dict["dataset_path"] == "./"
    assert yaml_dict["description"] == "A test metadata file."
    assert yaml_dict["date_downloaded"] == "2021-01-01"
    assert yaml_dict["download_source"] == "http://example.com"
    assert yaml_dict["data_content"] == "text"
    assert yaml_dict["content_field"] == "text"


def test_file_load(json_metadata, yaml_metadata, yml_metadata):
    for metadata in [json_metadata, yaml_metadata, yml_metadata]:
        path, filename = metadata
        metadata = TatmDataMetadata.from_file(path / filename)
        assert metadata.name == "test"
        assert metadata.dataset_path == str(path)
        assert metadata.description == "A test metadata file."
        assert metadata.date_downloaded == "2021-01-01"
        assert metadata.download_source == "http://example.com"
        assert metadata.data_content == DataContentType.TEXT
        assert metadata.content_field == "text"
        assert metadata.corpuses == []
        assert metadata.tokenized_info is None


def test_directory_load(json_metadata, yaml_metadata, yml_metadata):
    for metadata in [json_metadata, yaml_metadata, yml_metadata]:
        path, filename = metadata
        metadata = TatmDataMetadata.from_directory(str(path))
        assert metadata.name == "test"
        assert metadata.dataset_path == str(path)
        assert metadata.description == "A test metadata file."
        assert metadata.date_downloaded == "2021-01-01"
        assert metadata.download_source == "http://example.com"
        assert metadata.data_content == DataContentType.TEXT
        assert metadata.content_field == "text"
        assert metadata.corpuses == []
        assert metadata.tokenized_info is None


def test_interactive_creation(monkeypatch, tmp_path):

    output_types = ["", "json", "yaml"]
    content_types = ["", "text"]
    for output_type, content_type in [
        (o, c) for o in output_types for c in content_types
    ]:
        responses = iter(
            [
                output_type,
                tmp_path / f"metadata_test.{output_type if output_type else 'json'}",
                "test",
                "./",
                "A test metadata file.",
                "2021-01-01",
                "http://example.com",
                content_type,
                "text",
                "corpus1",
                "corpus2",
                "",
                "",
                "",
                "",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        create_metadata_interactive()
        if output_type in ["", "json"]:
            metadata = TatmDataMetadata.from_json(tmp_path / "metadata_test.json")
        else:
            metadata = TatmDataMetadata.from_yaml(tmp_path / "metadata_test.yaml")
        assert metadata.name == "test"
        assert metadata.dataset_path == "./"
        assert metadata.description == "A test metadata file."
        assert metadata.date_downloaded == "2021-01-01"
        assert metadata.download_source == "http://example.com"
        assert metadata.data_content == DataContentType.TEXT
        assert metadata.content_field == "text"
        assert metadata.corpuses == ["corpus1", "corpus2"]
        assert metadata.corpus_separation_strategy == "data_dirs"
        assert metadata.corpus_data_dir_parent is None
        assert metadata.tokenized_info is None


def test_tokenized_interactive_creation(monkeypatch, tmp_path):

    output_types = ["", "json", "yaml"]
    content_types = ["", "text"]
    for output_type, content_type in [
        (o, c) for o in output_types for c in content_types
    ]:
        responses = iter(
            [
                output_type,
                tmp_path / f"metadata_test.{output_type if output_type else 'json'}",
                "test",
                "./",
                "A test metadata file.",
                "2021-01-01",
                "http://example.com",
                content_type,
                "text",
                "corpus1",
                "corpus2",
                "",
                "",
                "",
                "y",
                "test_tokenizer",
                "tokenized",
                "",
                "uint8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        create_metadata_interactive()
        if output_type in ["", "json"]:
            metadata = TatmDataMetadata.from_file(tmp_path / "metadata_test.json")
        else:
            metadata = TatmDataMetadata.from_file(tmp_path / "metadata_test.yaml")
        assert metadata.name == "test"
        assert metadata.dataset_path == "./"
        assert metadata.description == "A test metadata file."
        assert metadata.date_downloaded == "2021-01-01"
        assert metadata.download_source == "http://example.com"
        assert metadata.data_content == DataContentType.TEXT
        assert metadata.content_field == "text"
        assert metadata.corpuses == ["corpus1", "corpus2"]
        assert metadata.corpus_separation_strategy == "data_dirs"
        assert metadata.corpus_data_dir_parent is None
        assert metadata.tokenized_info is not None
        assert metadata.tokenized_info.tokenizer == "test_tokenizer"
        assert metadata.tokenized_info.file_prefix == "tokenized"
        assert metadata.tokenized_info.dtype == "uint8"
        assert metadata.tokenized_info.file_extension == "bin"
