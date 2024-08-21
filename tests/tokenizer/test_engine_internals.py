import numpy as np
import ray
import tokenizers

from tatm.tokenizer.engine import (  # DataServer,; TokenWriter,
    ExampleMessage,
    TokenizerWorker,
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
