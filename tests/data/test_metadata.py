import json
import pathlib

import yaml

from tatm.data.metadata import (
    DatasetContentType,
    DatasetMetadata,
    create_metadata_interactive,
)


def test_json_load():
    filename = "tests/data/metadata_test.json"
    metadata = DatasetMetadata.from_json(filename)
    test_file_dir = pathlib.Path(filename).resolve().parent
    assert metadata.name == "test"
    assert metadata.dataset_path == str(test_file_dir)
    assert metadata.description == "A test metadata file."
    assert metadata.date_downloaded == "2021-01-01"
    assert metadata.download_source == "http://example.com"
    assert metadata.data_content == DatasetContentType.TEXT
    assert metadata.content_field == "text"


def test_yaml_load():
    filename = "tests/data/metadata_test.yaml"
    metadata = DatasetMetadata.from_yaml(filename)
    test_file_dir = pathlib.Path(filename).resolve().parent
    assert metadata.name == "test"
    assert metadata.dataset_path == str(test_file_dir)
    assert metadata.description == "A test metadata file."
    assert metadata.date_downloaded == "2021-01-01"
    assert metadata.download_source == "http://example.com"
    assert metadata.data_content == DatasetContentType.TEXT
    assert metadata.content_field == "text"


def test_json_save(tmp_path):
    metadata = DatasetMetadata(
        name="test",
        dataset_path="./",
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DatasetContentType.TEXT,
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
    metadata = DatasetMetadata(
        name="test",
        dataset_path="./",
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content=DatasetContentType.TEXT,
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
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        create_metadata_interactive()
        if output_type in ["", "json"]:
            metadata = DatasetMetadata.from_json(tmp_path / "metadata_test.json")
        else:
            metadata = DatasetMetadata.from_yaml(tmp_path / "metadata_test.yaml")
        assert metadata.name == "test"
        assert metadata.dataset_path == "./"
        assert metadata.description == "A test metadata file."
        assert metadata.date_downloaded == "2021-01-01"
        assert metadata.download_source == "http://example.com"
        assert metadata.data_content == DatasetContentType.TEXT
        assert metadata.content_field == "text"
        assert metadata.corpuses == ["corpus1", "corpus2"]
