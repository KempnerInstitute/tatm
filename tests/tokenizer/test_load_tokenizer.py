import pytest
import tokenizers

from tatm.tokenizer.utils import load_tokenizer


@pytest.fixture
def file_tokenizer(tmp_path):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer = tokenizers.Tokenizer.from_pretrained("t5-base")
    tokenizer.save(str(tokenizer_file))
    yield tokenizer_file
    return tokenizer_file


def test_load_file_tokenizer(file_tokenizer):
    tokenizer = load_tokenizer(str(file_tokenizer))
    assert tokenizer.get_vocab_size() == 32100  # Check the vocab size


def test_load_pretrained_tokenizer():
    tokenizer = load_tokenizer("t5-base")
    assert tokenizer.get_vocab_size() == 32100  # Check the vocab size


def test_load_invalid_tokenizer():
    with pytest.raises(ValueError):
        load_tokenizer("invalid")
