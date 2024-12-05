from importlib.metadata import version

import pytest
import tokenizers

from tatm.data.metadata import TatmDataMetadata
from tatm.tokenizer.metadata import write_metadata

tatm_version = version("tatm")


@pytest.fixture
def file_tokenizer(tmp_path):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer = tokenizers.Tokenizer.from_pretrained("t5-base")
    tokenizer.save(str(tokenizer_file))
    yield tokenizer_file
    return tokenizer_file


def test_write_metadata(tmp_path):
    write_metadata(
        "t5-base",
        str(tmp_path),
        "tokenized",
        data_description="Test dataset",
        parent_datasets=["test"],
    )
    # Check if the metadata file is created
    assert (tmp_path / "metadata.yaml").exists()
    # Check if the tokenizer file is created
    assert (tmp_path / "tokenizer.json").exists()
    metadata = TatmDataMetadata.from_yaml(tmp_path / "metadata.yaml")
    assert metadata.name == "tokenized"
    assert metadata.description == "Test dataset"
    assert metadata.dataset_path == str(tmp_path)
    assert metadata.tokenized_info.tokenizer == "t5-base"
    assert metadata.tokenized_info.file_prefix == str(tmp_path / "tokenized")
    assert metadata.tokenized_info.dtype == "uint16"
    assert metadata.tokenized_info.file_extension == "bin"  # Default value
    assert metadata.tokenized_info.vocab_size == 32100  # Check the vocab size
    assert (
        metadata.tokenized_info.tatm_version == tatm_version
    )  # Check the tokenizer name
    assert metadata.tokenized_info.parent_datasets == ["test"]


def test_write_metadata_with_file_tokenizer(file_tokenizer, tmp_path):
    write_metadata(
        str(file_tokenizer),
        str(tmp_path),
        "tokenized",
    )
    # Check if the metadata file is created
    assert (tmp_path / "metadata.yaml").exists()
    # Check if the tokenizer file is created
    assert (tmp_path / "tokenizer.json").exists()
    metadata = TatmDataMetadata.from_yaml(tmp_path / "metadata.yaml")
    assert metadata.name == "tokenized"
    assert metadata.description == "Tokenized dataset created using tatm"
    assert metadata.dataset_path == str(tmp_path)
    assert metadata.tokenized_info.tokenizer == str(file_tokenizer)
    assert metadata.tokenized_info.file_prefix == str(tmp_path / "tokenized")
    assert metadata.tokenized_info.dtype == "uint16"
    assert metadata.tokenized_info.file_extension == "bin"  # Default value
    assert metadata.tokenized_info.vocab_size == 32100  # Check the vocab size
    assert (
        metadata.tokenized_info.tatm_version == tatm_version
    )  # Check the tokenizer name
