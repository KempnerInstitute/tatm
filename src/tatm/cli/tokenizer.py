import os
import pathlib

import click
import ray

from tatm.tokenizer import Engine


@click.command()
@click.argument("datasets", nargs=-1)
@click.option(
    "--num-workers", default=1, help="Number of workers to use for tokenization"
)
@click.option(
    "--tokenizer", default="t5-base", help="Tokenizer to use for tokenization"
)
@click.option(
    "--output-dir",
    default=".",
    help="Output directory for tokenized data",
    type=click.Path(),
)
@click.option("--file-prefix", default="tokenized", help="Prefix for tokenized files")
def tokenize(datasets, num_workers, tokenizer, output_dir, file_prefix):
    os.makedirs(output_dir, exist_ok=True)

    ray.init()
    e = Engine(datasets, tokenizer, str(pathlib.Path(output_dir) / file_prefix))
    e.run_with_ray(num_workers)
    ray.shutdown()
