import dataclasses
import logging
import queue
import random
import time
from multiprocessing import (
    Queue,  # Used instead of default queue to allow for future mp based implementation
)
from pathlib import Path
from typing import List, Union

import numpy as np
import ray
import tokenizers

from tatm.data import TatmDataMetadata, get_data
from tatm.tokenizer.metadata import write_metadata
from tatm.tokenizer.utils import load_tokenizer
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
        data: List[Union[str, TatmDataMetadata]],
        seed: int = 2130,
        max_queue_size: int = 1024,
        log_level: str = logging.INFO,
    ):
        self.data = data
        self.datasets = [get_data(d) for d in data]
        self.seed = seed
        self.max_queue_size = max_queue_size
        self.initialized = False
        self.initialize()
        self.shutdown_flag = False
        configure_logging(log_level)
        self.debug_mode = log_level == logging.DEBUG
        if self.debug_mode:
            LOGGER.warning(
                "Running in debug mode. Note that this may impact performance as the queue interaction pattern changes to allow us to inspect the queue."
            )

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
            if not self.debug_mode:
                self.queue.put(example)
            else:
                while True:
                    try:
                        self.queue.put(example, block=False)
                        break
                    except queue.Full:
                        LOGGER.debug("DataServer queue is full")
                        time.sleep(0.1)

        while True:
            # Ensure that all workers receive a termination signal
            try:
                self.queue.put(None, block=False)
            except queue.Full:
                time.sleep(0.1)
            if self.shutdown_flag:
                break

    def next_item(self):
        if not self.debug_mode:
            return self.queue.get()
        else:
            while True:
                try:
                    return self.queue.get(block=False)
                except queue.Empty:
                    LOGGER.debug("DataServer queue is empty")
                    time.sleep(0.1)

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
        dtype: str = "uint32",
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
        self.debug_mode = log_level == logging.DEBUG
        if self.debug_mode:
            LOGGER.warning(
                "Running in debug mode. Note that this may impact performance as the queue interaction pattern changes to allow us to inspect the queue."
            )

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
        if not self.debug_mode:
            self.queue.put(data)
            return
        else:
            while True:
                try:
                    self.queue.put(data, block=False)
                    break
                except queue.Full:
                    LOGGER.debug("TokenWriter queue is full")
                    time.sleep(0.1)

    def get(self):
        if not self.debug_mode:
            return self.queue.get()
        else:
            while True:
                try:
                    return self.queue.get(block=False)
                except queue.Empty:
                    LOGGER.debug("TokenWriter queue is empty")
                    time.sleep(0.1)

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
            data = self.get()

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
        self.tokenizer = load_tokenizer(tokenizer)
        self.reset_threshold = reset_threshold
        self.documents_tokenized = 0
        configure_logging(log_level)
        self.debug_mode = log_level == logging.DEBUG
        if self.debug_mode:
            LOGGER.warning(
                "Running in debug mode. Note that this may impact performance as the queue interaction pattern changes to allow us to inspect the queue."
            )

    def process_example(self):
        example: ExampleMessage = ray.get(self.server.next_item.remote())
        if example is None:
            return False
        tokens = self.tokenizer.encode(example.data[example.content_field]).ids

        ray.get(self.writer.put.remote(np.array(tokens)))
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


class TokenizationEngine:
    def __init__(
        self,
        data: List[Union[str, TatmDataMetadata]],
        tokenizer: str,
        output_dir: str,
        file_prefix: str,
        dtype: str = "uint32",
        log_level: int = logging.INFO,
    ):
        """Object holding information needed to execute the tokenization process, along with methods to run it.

        Args:
            data: A list of either paths to Tatm Data collections or TatmDataMetadata objects.
            tokenizer: The name of a Hugging Face tokenizer or path to a tokenizer file.
            output_dir: The directory where the tokenized files will be saved.
            file_prefix: The prefix for the tokenized files.
            dtype: The numpy data type to use for the tokenized files. Defaults to "uint32".
            log_level: python logging level to use within Ray. Defaults to logging.INFO.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.dtype = dtype
        self.log_level = log_level

    def run_with_ray(self, num_workers: int = None):
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. Please initialize Ray before running the engine."
            )
        num_cpus = ray.cluster_resources()["CPU"]

        if not num_workers:
            num_workers = int(num_cpus) - 2
        if num_workers < 1:
            raise ValueError(
                "Number of workers must be greater than 0. Note that at least 3 CPUs must be available to ray if the number of workers is not being specified explicitly."
            )
        if num_workers > int(num_cpus) - 2:
            LOGGER.warning(
                f"Number of workers specified ({num_workers}) exceeds available CPUs ({num_cpus}). Setting number of workers to {int(num_cpus) - 2} to allow for reader and writer processes."
            )
            num_workers = int(num_cpus) - 2
        if num_workers < num_cpus - 2:
            LOGGER.warning(
                f"Number of workers specified ({num_workers}) is less than available CPUs ({num_cpus}). This may result in suboptimal performance."
            )

        LOGGER.info(
            f"Tokenizing data with 1 reader process, 1 writer process, and {num_workers} worker processes."
        )

        server = DataServer.options(max_concurrency=num_workers + 1).remote(
            self.data, log_level=self.log_level
        )
        writer = TokenWriter.options(max_concurrency=num_workers + 1).remote(
            str(Path(self.output_dir) / self.file_prefix),
            dtype=self.dtype,
            log_level=self.log_level,
        )
        workers = [
            TokenizerWorker.remote(
                self.tokenizer,
                server,
                writer,
                log_level=self.log_level,
            )
            for _ in range(num_workers)
        ]
        write_metadata(
            self.tokenizer, self.output_dir, self.file_prefix, dtype=self.dtype
        )
        s = server.run.remote()
        writer.run.remote()
        ray.get([worker.run.remote() for worker in workers])
        writer.put.remote(None)  # Signal the writer to shutdown
        # Shutdown the server
        server.shutdown.remote()
        ray.get(s)  # block until server is done
