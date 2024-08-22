from typing import Union

import numpy as np
import ray
import tokenizers

from tatm.tokenizer.engine import (  # DataServer,; TokenWriter,
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
    ray.init(local_mode=True)  # Initialize Ray in local mode for testing
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


def test_data_server():
    # Test that the data server returns the correct data
    # Can't run the run method due to the infinite loop causing blocking/the test to hang
    ray.init(local_mode=True, num_cpus=4)  # Initialize Ray in local mode for testing
    server = DataServer.options(max_concurrency=4).remote(
        ["tests/data/json_dataset"], max_queue_size=8
    )
    example = ray.get(server.get_example.remote())
    assert example.data[example.content_field] == "hello world"
    for _ in range(7):
        example = ray.get(server.get_example.remote())
    assert example is None
    ray.get(server.shutdown.remote())
    ray.shutdown()  # Clean up Ray resources


def test_token_writer(tmp_path):
    # Test that the token writer correctly writes data to a file
    ray.init(local_mode=True)  # Initialize Ray in local mode for testing
    writer = TokenWriter.remote(str(tmp_path / "test"))
    data = np.array([1, 2, 3, 4, 5], dtype="uint16")
    writer.put.remote(data)
    writer.put.remote(None)
    ray.get(writer.run.remote())
    x = np.memmap(str(tmp_path / "test_0.bin"), dtype="uint16", mode="r")
    assert np.all(x[0:5] == data)
    ray.shutdown()  # Clean up Ray resources
