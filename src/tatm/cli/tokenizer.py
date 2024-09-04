import logging
import os
import pathlib

import click
import ray

from tatm.tokenizer import Engine


@click.command()
@click.argument("datasets", nargs=-1)
@click.option(
    "--num-workers", default=None, help="Number of workers to use for tokenization"
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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--file-prefix", default="tokenized", help="Prefix for tokenized files")
def tokenize(datasets, num_workers, tokenizer, output_dir, file_prefix, verbose):
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    os.makedirs(output_dir, exist_ok=True)

    ray.init()
    e = Engine(
        datasets,
        tokenizer,
        str(pathlib.Path(output_dir) / file_prefix),
        log_level=log_level,
    )
    e.run_with_ray(num_workers)
    ray.shutdown()
