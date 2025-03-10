import logging
import os

import click
import ray

from tatm.cli.utils import configure_cli_logging
from tatm.tokenizer import TokenizationEngine


@click.command()
@click.argument("datasets", nargs=-1)
@click.option(
    "--num-workers",
    default=None,
    help="Number of workers to use for tokenization",
    type=int,
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
@click.option(
    "--token-dtype", default="uint32", help="Numpy data type for tokenized files"
)
def tokenize(
    datasets, num_workers, tokenizer, output_dir, file_prefix, verbose, token_dtype
):
    """Tokenize a dataset using the tatm ray based tokenization engine.
    If running in a cluster environment, it is recommended to use this command
    in conjunction with the `run` command to submit the tokenization job to the cluster.

    This command will tokenize the input datasets using the specified tokenizer and output the tokenized data to the specified output directory as
    a series of binary files. The number of workers to use for tokenization can be specified using the `--num-workers` option. If not specified, the number of workers will be determined by the number of available CPUs
    to the ray cluster.

    Arguments:
    DATASETS: Paths to the datasets to tokenize. This command can accept multiple datasets to tokenize. All datasets are expected
    to have a tatm metadata file associated with them.
    """
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    configure_cli_logging(log_level)
    os.makedirs(output_dir, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        logging_config=ray.LoggingConfig(
            encoding="TEXT",
            log_level=logging.getLevelName(log_level),
            additional_log_standard_attrs=["name"],
        ),
    )  # Looks to RAY_ADDRESS env var for connection to remote cluster, or starts a local cluster

    e = TokenizationEngine(
        datasets,
        tokenizer,
        output_dir,
        file_prefix,
        log_level=log_level,
        dtype=token_dtype,
    )
    e.run_with_ray(num_workers)
    ray.shutdown()
