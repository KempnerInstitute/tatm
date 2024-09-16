import numpy as np
import pytest

from tatm.data import TatmMemmapDataset, get_dataset
from tatm.data.datasets import TokenMemMapArray
from tatm.tokenizer.metadata import write_metadata


@pytest.fixture()
def sample_dataset(tmp_path):
    for i in range(10):
        data = np.memmap(
            tmp_path / f"test_{i}.bin", dtype="uint16", mode="w+", shape=(1000,)
        )
        data[:] = i * 1000 + np.arange(1000)
        data.flush()
        del data
    write_metadata("t5-base", "test", str(tmp_path))
    yield (tmp_path, "test")

    for i in range(10):
        (tmp_path / f"test_{i}.bin").unlink()


def test_memmap_array(sample_dataset):
    test_file = f"{str(sample_dataset[0]/sample_dataset[1])}_0.bin"
    memmap_array = TokenMemMapArray(test_file, 100, "uint16", True)
    assert len(memmap_array) == 10
    assert np.all(memmap_array[0] == np.arange(100))


def test_memmap_dataset(sample_dataset):
    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]), 100, "uint16"
    )
    assert len(dataset) == 100
    assert np.all(dataset[0] == np.arange(100))
    assert np.all(dataset[20] == np.arange(100) + 2000)
    assert dataset.num_files() == 10
    assert dataset.num_tokens() == 100 * 100


def test_memmap_dataset_from_metadata(sample_dataset):
    dataset = get_dataset(str(sample_dataset[0]), context_length=100)
    assert len(dataset) == 100
    assert np.all(dataset[0] == np.arange(100))
    assert np.all(dataset[20] == np.arange(100) + 2000)
    assert dataset.num_files() == 10
    assert dataset.num_tokens() == 100 * 100
