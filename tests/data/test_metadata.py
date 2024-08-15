import pathlib
import json
import yaml
from tatm.data.metadata import Metadata


def test_json_load():
    filename = "tests/data/metadata_test.json"
    metadata = Metadata.from_json(filename)
    test_file_dir = pathlib.Path(filename).resolve().parent
    assert metadata.name == "test"
    assert metadata.dataset_path == str(test_file_dir)
    assert metadata.description == "A test metadata file."
    assert metadata.date_downloaded == "2021-01-01"
    assert metadata.download_source == "http://example.com"
    assert metadata.data_content == "text"
    assert metadata.content_field == "text"


def test_yaml_load():
    filename = "tests/data/metadata_test.yaml"
    metadata = Metadata.from_yaml(filename)
    test_file_dir = pathlib.Path(filename).resolve().parent
    assert metadata.name == "test"
    assert metadata.dataset_path == str(test_file_dir)
    assert metadata.description == "A test metadata file."
    assert metadata.date_downloaded == "2021-01-01"
    assert metadata.download_source == "http://example.com"
    assert metadata.data_content == "text"
    assert metadata.content_field == "text"


def test_json_save(tmp_path):
    metadata = Metadata(
        name="test",
        dataset_path="./",
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content="text",
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
    metadata = Metadata(
        name="test",
        dataset_path="./",
        description="A test metadata file.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content="text",
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
