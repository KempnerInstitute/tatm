import pathlib
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
