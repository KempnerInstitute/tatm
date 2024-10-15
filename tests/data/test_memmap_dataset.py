import numpy as np
import pytest
from torch.utils.data import DataLoader

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
    write_metadata("t5-base", str(tmp_path), "test")
    yield (tmp_path, "test")

    for i in range(10):
        (tmp_path / f"test_{i}.bin").unlink()


def test_memmap_array(sample_dataset):
    test_file = f"{str(sample_dataset[0]/sample_dataset[1])}_0.bin"
    memmap_array = TokenMemMapArray(test_file, 100, "uint16", True)
    assert len(memmap_array) == 10
    assert np.all(memmap_array[0] == np.arange(100))

    memmap_array_not_divisible = TokenMemMapArray(test_file, 9, "uint16", True)
    assert len(memmap_array_not_divisible) == 112
    assert len(memmap_array_not_divisible[111]) == 9
    assert np.all(
        memmap_array_not_divisible[111] == np.array([999, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_memmap_dataset(sample_dataset):
    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]), 100, "uint16"
    )
    assert len(dataset) == 100
    assert np.all(dataset[0]["token_ids"] == np.arange(100))
    assert np.all(dataset[20]["token_ids"] == np.arange(100) + 2000)
    assert np.all(dataset[-1]["token_ids"] == np.arange(100) + 9900)
    assert isinstance(dataset[0]["token_ids"], np.ndarray)
    assert not isinstance(dataset[0]["token_ids"], np.memmap)
    assert isinstance(dataset[0]["document_ids"], np.ndarray)
    assert not isinstance(dataset[0]["document_ids"], np.memmap)
    assert dataset.num_files() == 10
    assert dataset.num_tokens() == 100 * 100


def test_memmap_dataset_from_metadata(sample_dataset):
    dataset = get_dataset(str(sample_dataset[0]), context_length=100)
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


def test_memmap_dataset_docid_options(sample_dataset):
    # Test that the dataset doc id options raise errors when they should
    with pytest.raises(ValueError):
        _ = TatmMemmapDataset(
            str(sample_dataset[0] / sample_dataset[1]),
            100,
            "uint16",
            chunked=False,
            create_doc_ids=False,
            create_doc_mask=True,
        )

    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]),
        100,
        "uint16",
        chunked=True,
        create_doc_ids=True,
        create_doc_mask=True,
    )
    assert len(dataset) == 100
    assert np.all(dataset[0]["token_ids"] == np.arange(100))
    assert np.all(dataset[20]["token_ids"] == np.arange(100) + 2000)
    assert isinstance(dataset[0]["token_ids"], np.ndarray)
    assert not isinstance(dataset[0]["token_ids"], np.memmap)
    assert isinstance(dataset[0]["document_ids"], np.ndarray)
    assert not isinstance(dataset[0]["document_ids"], np.memmap)
    assert len(dataset[0]["document_ids"]) == 100
    assert isinstance(dataset[0]["document_mask"], np.ndarray)
    assert not isinstance(dataset[0]["document_mask"], np.memmap)
    assert dataset[0]["document_mask"].shape == (100, 100)


def test_dataloader_integration(sample_dataset):
    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]), 100, "uint16", create_doc_mask=True
    )
    dl = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        collate_fn=TatmMemmapDataset.torch_collate_fn,
    )

    batch = next(iter(dl))
    assert batch["token_ids"].shape == (10, 100)
    assert batch["document_ids"].shape == (10, 100)
    assert batch["document_masks"].shape == (10, 100, 100)

    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]), 100, "uint16", create_doc_ids=False
    )
    dl = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        collate_fn=TatmMemmapDataset.torch_collate_fn,
    )
    batch = next(iter(dl))
    assert "document_ids" not in batch

    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]), 100, "uint16"
    )
    dl = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        collate_fn=TatmMemmapDataset.torch_collate_fn,
    )
    batch = next(iter(dl))
    assert "document_masks" not in batch
