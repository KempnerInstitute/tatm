from pathlib import Path

import click

import tatm.compute.run
from tatm.config import load_config


def parse_config_opts(opts, validate=True):
    """Parse passed in configuration options. Determine if they are files or overrides.
    Assumes that if a string contains an `=` it is an override.

    Args:
        opts (List[str]): List of configuration options or files.
        validate (bool, optional): Should config files be checked for existence. Defaults to True. Primarily for testing.

    Raises:
        FileNotFoundError: Raised if a config file is not found and validate is True.

    Returns:
        Tuple[List[str]]: Returns a tuple of length 2. The first element is a list of config files, the second is a list of overrides.
    """
    files, overrides = [], []
    for opt in opts:
        if "=" in opt:
            overrides.append(opt)
        else:
            file_path = Path(opt)
            if validate and not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {opt}")
            files.append(opt)
    return files, overrides


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("wrapped_command", nargs=-1)
@click.option(
    "--config",
    "-c",
    default=None,
    help=(
        "Path to configuration file or specific configuration settings. First, all config"
        " files are merged, then all config options are merged. Command line options override config file options."
        " CLI config options should be in the format `field.subfield=value`. For list types, use dotlist notation"
        " (e.g. `field.subfield=[1,2,3]`). Note that this will override any list values in the config file."
        " Also note that this may cause issues with your shell, so be sure to quote the entire argument."
    ),
    multiple=True,
)
@click.option(
    "--nodes",
    "-N",
    default=None,
    type=int,
    help="Number of nodes to use for wrapped command.",
)
@click.option(
    "--cpus-per-task",
    "--cpus",
    default=None,
    type=int,
    help="Number of CPUs to use per task.",
)
@click.option("--submit-script", default=None, help="Path to submit script to create.")
@click.option("--time-limit", default=None, help="Time limit for the job.")
@click.option("--memory", default=None, help="Memory to allocate for the job.")
@click.option("--gpus-per-node", default=None, help="Number of GPUs to use per node.")
@click.option("--constraints", default=None, help="Constraints for the job.")
@click.option(
    "--submit/--no-submit",
    default=True,
    help="Submit the job after creating the submit script. Set to False to only create the submit script.",
)
def run(**kwargs):
    print(kwargs)
    config = kwargs.pop("config")
    wrapped_command = kwargs.pop("wrapped_command")
    files, overrides = parse_config_opts(config)
    cfg = load_config(files, overrides)

    options = tatm.compute.run.TatmRunOptions(**kwargs)
    print(options)
    print(cfg)
    result = tatm.compute.run.run(cfg, options, wrapped_command)
    if not kwargs["submit"]:
        print(" ".join(result))
