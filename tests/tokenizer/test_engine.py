import gc

import numpy as np
import ray
import ray.util.state
import tokenizers

import tatm.data
from tatm.tokenizer import Engine


def test_ray_run(tmp_path):
    # Initialize Ray
    ray.init(num_cpus=4)
    dataset = tatm.data.get_dataset("tests/data/json_dataset")
    tokenizer = tokenizers.Tokenizer.from_pretrained("t5-base")

    # Create an instance of the Engine
    engine = Engine(["tests/data/json_dataset"], "t5-base", str(tmp_path / "test"))

    # Run the engine with some input
    engine.run_with_ray(num_workers=1)
    gc.collect()

    first_example = next(iter(dataset))
    tokenized = tokenizer.encode(first_example[dataset.metadata.content_field]).ids

    engine_result = np.memmap(str(tmp_path / "test_0.bin"), dtype="uint16", mode="r")
    assert np.array_equal(engine_result[0 : len(tokenized)], tokenized)

    actors = ray.util.state.list_actors()
    assert (
        len([x for x in actors if x.state == "ALIVE"]) == 0
    )  # Ensure all actors are cleaned up

    # Clean up
    ray.shutdown()
