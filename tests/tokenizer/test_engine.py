import gc
import logging
import time

import numpy as np
import ray
import ray.util.state
import tokenizers

import tatm.data
from tatm.tokenizer import TokenizationEngine


def test_ray_run(tmp_path):
    """Integration test for the Engine class with Ray."""
    # Initialize Ray
    ray.init(num_cpus=4, ignore_reinit_error=True)
    dataset = tatm.data.get_data("tests/data/json_data")
    tokenizer = tokenizers.Tokenizer.from_pretrained("t5-base")

    # Create an instance of the Engine
    engine = TokenizationEngine(
        ["tests/data/json_data"], "t5-base", str(tmp_path), "test"
    )

    # Run the engine with some input
    engine.run_with_ray(num_workers=1)
    time.sleep(1)  # Wait for the workers to finish
    gc.collect()

    first_example = next(iter(dataset))
    tokenized = tokenizer.encode(first_example[dataset.metadata.content_field]).ids

    actors = ray.util.state.list_actors()
    assert len(actors) == 3  # Ensure only 1 worker is created
    assert (
        len([x for x in actors if x.state == "ALIVE"]) == 0
    )  # Ensure all actors are cleaned up

    # Clean up ray
    ray.shutdown()

    engine_result = np.memmap(str(tmp_path / "test_0.bin"), dtype="uint16", mode="r")
    assert np.array_equal(engine_result[0 : len(tokenized)], tokenized)


def test_ray_no_specified_workers(tmp_path):
    """Integration test for the Engine class with Ray testing that number of worker logic setting is correct."""
    # Initialize Ray
    ray.init(num_cpus=3, ignore_reinit_error=True)
    dataset = tatm.data.get_data("tests/data/json_data")
    tokenizer = tokenizers.Tokenizer.from_pretrained("t5-base")

    # Create an instance of the Engine
    engine = TokenizationEngine(
        ["tests/data/json_data"], "t5-base", str(tmp_path), "test"
    )

    # Run the engine with some input
    engine.run_with_ray(num_workers=None)
    time.sleep(1)
    gc.collect()

    first_example = next(iter(dataset))
    tokenized = tokenizer.encode(first_example[dataset.metadata.content_field]).ids

    actors = ray.util.state.list_actors()
    assert len(actors) == 3  # Ensure 4 workers are created
    assert (
        len([x for x in actors if x.state == "ALIVE"]) == 0
    )  # Ensure all actors are cleaned up

    # Clean up ray
    ray.shutdown()

    engine_result = np.memmap(str(tmp_path / "test_0.bin"), dtype="uint16", mode="r")
    assert np.array_equal(engine_result[0 : len(tokenized)], tokenized)


def test_ray_too_many_workers(tmp_path):
    # Initialize Ray
    ray.init(num_cpus=3, ignore_reinit_error=True)
    dataset = tatm.data.get_data("tests/data/json_data")
    tokenizer = tokenizers.Tokenizer.from_pretrained("t5-base")

    # Create an instance of the Engine
    engine = TokenizationEngine(
        ["tests/data/json_data"], "t5-base", str(tmp_path), "test"
    )

    # Run the engine with some input
    engine.run_with_ray(num_workers=16)
    time.sleep(1)
    gc.collect()

    first_example = next(iter(dataset))
    tokenized = tokenizer.encode(first_example[dataset.metadata.content_field]).ids

    actors = ray.util.state.list_actors()
    assert len(actors) == 3  # Ensure 4 workers are created
    assert (
        len([x for x in actors if x.state == "ALIVE"]) == 0
    )  # Ensure all actors are cleaned up

    # Clean up ray
    ray.shutdown()

    engine_result = np.memmap(str(tmp_path / "test_0.bin"), dtype="uint16", mode="r")
    assert np.array_equal(engine_result[0 : len(tokenized)], tokenized)


def test_ray_run_in_debug_mode(tmp_path):

    # Initialize Ray
    ray.init(num_cpus=4, ignore_reinit_error=True)

    dataset = tatm.data.get_data("tests/data/json_data")
    tokenizer = tokenizers.Tokenizer.from_pretrained("t5-base")

    # Create an instance of the Engine
    engine = TokenizationEngine(
        ["tests/data/json_data"],
        "t5-base",
        str(tmp_path),
        "test",
        log_level=logging.DEBUG,
    )

    # Run the engine with some input
    engine.run_with_ray(num_workers=1)
    time.sleep(1)
    gc.collect()

    first_example = next(iter(dataset))
    tokenized = tokenizer.encode(first_example[dataset.metadata.content_field]).ids

    actors = ray.util.state.list_actors()
    assert len(actors) == 3  # Ensure only 1 worker is created
    assert (
        len([x for x in actors if x.state == "ALIVE"]) == 0
    )  # Ensure all actors are cleaned up

    # Clean up ray
    ray.shutdown()

    engine_result = np.memmap(str(tmp_path / "test_0.bin"), dtype="uint16", mode="r")
    assert np.array_equal(engine_result[0 : len(tokenized)], tokenized)
