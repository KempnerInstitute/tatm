import json
import logging
import os
import pathlib
from typing import Union

import numpy as np
import pytest
import ray
import tokenizers

from tatm.data.metadata import TatmDataMetadata
from tatm.tokenizer.engine import (
    DataServer,
    ExampleMessage,
    TokenizerWorker,
    TokenWriter,
)


@ray.remote
class MockDataServer:
    def __init__(self, size=1):
        self.size = size
        self.index = 0

    def next_item(self):
        if self.index < self.size:
            self.index += 1
            out = ExampleMessage({"text": "hello world"}, content_field="text")
            return out
        else:
            return None


@ray.remote
class MockTokenWriter:
    def __init__(self):
        self.data = []

    def put(self, data):
        self.data.append(data)

    def get(self):
        return self.data


class MockWorker:
    def __init__(
        self,
        server: Union[MockDataServer, DataServer],
        writer: Union[MockTokenWriter, TokenWriter],
    ):
        self.server = server
        self.writer = writer


def test_worker_process_example():
    ray.init(
        local_mode=True, ignore_reinit_error=True
    )  # Initialize Ray in local mode for testing
    server = MockDataServer.remote(size=2)
    writer = MockTokenWriter.remote()
    worker = TokenizerWorker.remote("t5-base", server, writer, reset_threshold=1)
    worker.run.remote()
    tokenized_data = (
        tokenizers.Tokenizer.from_pretrained("t5-base").encode("hello world").ids
    )
    data = ray.get(writer.get.remote())
    print(data)
    assert len(data) == 2
    assert np.all(data[0] == tokenized_data)
    assert np.all(data[1] == tokenized_data)

    ray.shutdown()  # Clean up Ray resources


def test_token_writer(tmp_path):
    # Test that the token writer correctly writes data to a file
    ray.init(
        local_mode=True, ignore_reinit_error=True
    )  # Initialize Ray in local mode for testing
    writer = TokenWriter.remote(str(tmp_path / "test"))
    data = np.array([1, 2, 3, 4, 5], dtype="uint32")
    writer.put.remote(data)
    writer.put.remote(None)
    ray.get(writer.run.remote())
    x = np.memmap(str(tmp_path / "test_0.bin"), dtype="uint32", mode="r")
    assert np.all(x[0:5] == data)

    # Test that we run in debug mode
    writer = TokenWriter.remote(str(tmp_path / "test"), log_level=logging.DEBUG)
    data = np.array([1, 2, 3, 4, 5], dtype="uint32")
    writer.put.remote(data)
    writer.put.remote(None)
    ray.get(writer.run.remote())
    x = np.memmap(str(tmp_path / "test_0.bin"), dtype="uint32", mode="r")
    assert np.all(x[0:5] == data)

    os.remove(str(tmp_path / "test_0.bin"))
    ray.shutdown()  # Clean up Ray resources


class ExampleDatasetFactory:
    def __init__(self, parent_dir: pathlib.Path):
        self.parent_dir = parent_dir
        self.dataset_count = 0

    def create_dataset(self, num_examples) -> pathlib.Path:
        dataset_dir = self.parent_dir / f"dataset_{self.dataset_count}"
        dataset_dir.mkdir()
        with open(dataset_dir / "data.json", "w") as f:
            for i in range(num_examples):
                f.write(json.dumps({"text": str(i)}) + "\n")
        metadata = TatmDataMetadata(
            name=f"dataset_{self.dataset_count}",
            dataset_path=str(dataset_dir),
            description="Dataset of numbered examples, intended for concurrency testing",
            date_downloaded="2021-01-01",
            download_source="http://example.com",
            data_content="text",
            content_field="text",
        )
        metadata.to_yaml(dataset_dir / "metadata.yaml")
        self.dataset_count += 1
        return dataset_dir

    def __getitem__(self, index):
        if index <= self.dataset_count:
            raise IndexError("Index out of range")
        return self.parent_dir / f"dataset_{index}"


@pytest.fixture
def example_dataset_factory(tmp_path):
    factory = ExampleDatasetFactory(tmp_path)

    yield factory


class TestDataServerMethods:

    def test_list_sources(self):
        ray.init(local_mode=True, num_cpus=4, ignore_reinit_error=True)
        server = DataServer.remote(
            ["tests/data/json_data", "tests/data/json_corpus_data:primary"],
            max_queue_size=8,
        )
        sources = ray.get(server.list_source_datasets.remote())
        assert "test" in sources
        assert "json_corpus_data:primary" in sources

    def test_data_server(self):
        # Test that the data server returns the correct data
        # Can't run the run method due to the infinite loop causing blocking/the test to hang
        ray.init(
            local_mode=True, num_cpus=4, ignore_reinit_error=True
        )  # Initialize Ray in local mode for testing
        server = DataServer.options(max_concurrency=4).remote(
            ["tests/data/json_data"], max_queue_size=8
        )
        example = ray.get(server.get_example.remote())
        assert example.data[example.content_field] == "hello world"
        for _ in range(7):
            example = ray.get(server.get_example.remote())
        assert example is None
        ray.get(server.shutdown.remote())

        # Test that we run in debug mode
        server = DataServer.options(max_concurrency=4).remote(
            ["tests/data/json_data"], max_queue_size=2, log_level=logging.DEBUG
        )
        example = ray.get(server.get_example.remote())
        assert example.data[example.content_field] == "hello world"
        for _ in range(7):
            example = ray.get(server.get_example.remote())
        assert example is None
        ray.get(server.shutdown.remote())

        ray.shutdown()  # Clean up Ray resources

    def test_data_server_threaded_collisions(self, example_dataset_factory):
        ray.init(num_cpus=4, ignore_reinit_error=True)
        dataset = example_dataset_factory.create_dataset(100)
        server = DataServer.options(max_concurrency=4).remote(
            [str(dataset)], max_queue_size=100
        )
        server.run.remote()
        counter = 0
        results = set()
        example = ray.get(server.next_item.remote())
        while example is not None:
            counter += 1
            results.add(int(example.data[example.content_field]))
            example = ray.get(server.next_item.remote())
        print(f"Unique Examples seen: {len(results)}")
        print(f"Total Examples seen: {counter}")
        assert len(results) == counter  # All examples should be unique
        ray.get(server.shutdown.remote())
        ray.shutdown()

    @pytest.mark.parametrize("num_workers", [1, 2, 4, 8, 16])
    def test_data_server_multiple_workers(self, example_dataset_factory, num_workers):
        ray.init(num_cpus=4, ignore_reinit_error=True)
        dataset = example_dataset_factory.create_dataset(10000)
        server = DataServer.options(max_concurrency=8 + num_workers).remote(
            [str(dataset)], max_queue_size=100
        )
        for i in range(num_workers):
            server.run.remote()
        counter = 0
        example = ray.get(server.next_item.remote())
        while example is not None:
            counter += 1
            example = ray.get(server.next_item.remote())
        print(f"Total Examples seen: {counter}")
        assert counter == 10000
        ray.get(server.shutdown.remote())
        ray.shutdown()

    @pytest.mark.parametrize("num_workers", [1, 2, 4, 8, 16])
    def test_data_server_multiple_sets(self, example_dataset_factory, num_workers):
        ray.init(num_cpus=4, ignore_reinit_error=True)
        dataset1 = example_dataset_factory.create_dataset(10000)
        dataset2 = example_dataset_factory.create_dataset(100)
        server = DataServer.options(max_concurrency=8 + num_workers).remote(
            [str(dataset1), str(dataset2)], max_queue_size=100
        )
        for i in range(num_workers):
            server.run.remote()
        counter = 0
        example = ray.get(server.next_item.remote())
        while example is not None:
            counter += 1
            example = ray.get(server.next_item.remote())
        print(f"Total Examples seen: {counter}")
        assert counter == 10100
        ray.get(server.shutdown.remote())
        ray.shutdown()
