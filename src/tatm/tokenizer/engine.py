import dataclasses
import logging
import queue
import random
import time
from multiprocessing import (
    Queue,  # Used instead of default queue to allow for future mp based implementation
)
from typing import List, Union

import numpy as np
import ray
import tokenizers

from tatm.data import DatasetMetadata, get_dataset
from tatm.utils import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ExampleMessage:
    data: dict
    content_field: str


@ray.remote
class DataServer:
    def __init__(
        self,
        data: List[Union[str, DatasetMetadata]],
        seed: int = 2130,
        max_queue_size: int = 1024,
        log_level: str = logging.INFO,
    ):
        self.data = data
        self.datasets = [get_dataset(d) for d in data]
        self.seed = seed
        self.max_queue_size = max_queue_size
        self.initialized = False
        self.initialize()
        self.shutdown_flag = False
        configure_logging(log_level)

    def initialize(self):
        self.dataset_iters = [iter(dataset) for dataset in self.datasets]
        self.rng = random.Random(self.seed)
        self.queue = Queue(maxsize=self.max_queue_size)
        self.initialized = True
        self.done = False

    def get_example(self):
        if len(self.dataset_iters) == 0:
            LOGGER.info("No datasets available to iterate over.")
            self.done = True
            self.initialized = False
            return None
        if self.done:
            return None
        if not self.initialized:
            raise RuntimeError("DataServer not initialized. Call 'initialize' first.")
        dataset_idx = self.rng.randint(0, len(self.datasets) - 1)
        try:
            example = next(self.dataset_iters[dataset_idx])
            content_field = self.datasets[dataset_idx].metadata.content_field
            return ExampleMessage(data=example, content_field=content_field)
        except StopIteration:
            self.dataset_iters.pop(dataset_idx)
            return self.get_example()

    def run(self):
        while not self.done:
            example = self.get_example()
            if example is None:
                LOGGER.info("No more examples to fetch.")
                break
            self.queue.put(example)

        while True:
            # Ensure that all workers receive a termination signal
            try:
                self.queue.put(None, block=False)
            except queue.Full:
                time.sleep(0.1)
            if self.shutdown_flag:
                break

    def next_item(self):
        return self.queue.get()

    def shutdown(self):
        self.shutdown_flag = True
        self.done = True


@ray.remote
class TokenWriter:
    def __init__(
        self,
        file_prefix: str,
        max_file_size: int = 1024 * 1024 * 1024,
        max_queue_size: int = 1024,
        dtype: str = "uint16",
        log_level: str = logging.INFO,
    ):
        self.file_prefix = file_prefix
        self.max_file_size = max_file_size
        self.dtype = dtype
        self.current_array = None
        self.position = 0
        self.file_id = 0
        self.num_written = 0
        self.shutdown = False
        self.queue = Queue(maxsize=max_queue_size)
        configure_logging(log_level)

    def create_new_file(self):
        self.current_array = np.memmap(
            f"{self.file_prefix}_{self.file_id}.bin",
            dtype=self.dtype,
            mode="w+",
            shape=(self.max_file_size,),
        )
        self.position = 0

    def complete_current_file(self):
        if self.current_array is not None:
            LOGGER.info(f"Flushing data to {self.file_prefix}_{self.file_id}.bin")
            self.current_array.flush()
            self.current_array = None
            self.file_id += 1

    def put(self, data: np.ndarray):
        self.queue.put(data)

    def write(self, data: np.ndarray):
        if self.current_array is None:
            self.create_new_file()
        if self.position + len(data) > self.max_file_size:
            self.complete_current_file()
            self.create_new_file()
        self.current_array[self.position : self.position + len(data)] = data
        self.position += len(data)
        self.num_written += 1
        if self.num_written % 1000 == 0:
            if self.num_written % 10000 == 0:
                LOGGER.info(
                    f"Written {self.num_written} items to {self.file_prefix}_{self.file_id}.bin"
                )
            else:
                LOGGER.debug(
                    f"Written {self.num_written} items to {self.file_prefix}_{self.file_id}.bin"
                )

    def run(self):
        while not self.shutdown:
            data = self.queue.get()
            if data is None:
                LOGGER.info("Received shutdown signal.")
                break
            self.write(data)
        self.close()

    def close(self):
        if self.current_array is not None:
            self.complete_current_file()
        self.shutdown = True


@ray.remote
class TokenizerWorker:
    """Worker that requests data from the DataServer and sends tokenized examples to the TokenWriter."""

    def __init__(
        self,
        tokenizer: str,
        server: DataServer,
        writer: TokenWriter,
        reset_threshold: int = 10000,
        log_level: str = logging.INFO,
    ):
        self.server = server
        self.writer = writer
        self.tokenizer_name = tokenizer
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer)
        self.reset_threshold = reset_threshold
        self.documents_tokenized = 0
        configure_logging(log_level)

    def process_example(self):
        if ray.is_initialized():
            example: ExampleMessage = ray.get(self.server.next_item.remote())
        else:
            example: ExampleMessage = self.server.next_item()
        if example is None:
            return False
        tokens = self.tokenizer.encode(example.data[example.content_field]).ids
        if ray.is_initialized():
            ray.get(self.writer.put.remote(np.array(tokens)))
        else:
            self.writer.put(np.array(tokens))
        return True

    def run(self):
        while True:
            if not self.process_example():
                break
            self.documents_tokenized += 1
            if self.documents_tokenized % self.reset_threshold == 0:
                LOGGER.info(
                    f"Processed {self.documents_tokenized} documents. Resetting tokenizer to avoid memory leak issues."
                )
                self.reset_tokenizer()
        LOGGER.info(f"Processed {self.documents_tokenized} documents.")

    def reset_tokenizer(self):
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(self.tokenizer_name)


class Engine:
    def __init__(
        self,
        data: List[Union[str, DatasetMetadata]],
        tokenizer: str,
        file_prefix: str,
        log_level: str = logging.INFO,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.file_prefix = file_prefix
        self.log_level = log_level

    def run_with_ray(self, num_workers=2):
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. Please initialize Ray before running the engine."
            )
        server = DataServer.options(max_concurrency=num_workers + 1).remote(
            self.data, log_level=self.log_level
        )
        writer = TokenWriter.options(max_concurrency=num_workers + 1).remote(
            self.file_prefix,
            log_level=self.log_level,
        )
        workers = [
            TokenizerWorker.remote(
                self.tokenizer, server, writer, log_level=self.log_level
            )
            for _ in range(num_workers)
        ]
        s = server.run.remote()
        writer.run.remote()
        ray.get([worker.run.remote() for worker in workers])
        writer.put.remote(None)  # Signal the writer to shutdown
        # Shutdown the server
        server.shutdown.remote()
        ray.get(s)  # block until server is done
