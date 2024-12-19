from pathlib import Path
from typing import List, Tuple

import click

import tatm.compute.run
from tatm.config import load_config, _set_cli_config_files, _set_cli_config_overrides


def parse_config_opts(opts: List[str], validate=True) -> Tuple[List[str], List[str]]:
    """Parse passed in configuration options. Determine if they are files or overrides.
    Assumes that if a string contains an `=` it is an override.

    Args:
        opts: List of configuration options or files.
        validate: Should config files be checked for existence. Defaults to True. Primarily for testing.

    Raises:
        FileNotFoundError: Raised if a config file is not found and validate is True.

    Returns:
        Returns a tuple of length 2. The first element is a list of config files, the second is a list of overrides.
    """
    global CLI_CONFIG_FILES, CLI_CONFIG_OVERRIDES
    files, overrides = [], []
    for opt in opts:
        if "=" in opt:
            overrides.append(opt)
        else:
            file_path = Path(opt)
            if validate and not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {opt}")
            files.append(opt)
    _set_cli_config_files(files)
    _set_cli_config_overrides(overrides)
    return files, overrides


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("wrapped_command", nargs=-1)
@click.option(
    "--config",
    "--conf",
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
    "-c",
    default=None,
    type=int,
    help="Number of CPUs to use per task.",
)
@click.option("--submit-script", default=None, help="Path to submit script to create.")
@click.option("--time-limit", default=None, help="Time limit for the job.")
@click.option("--memory", "--mem", default=None, help="Memory to allocate for the job.")
@click.option("--gpus-per-node", default=None, help="Number of GPUs to use per node.")
@click.option("--constraints", default=None, help="Constraints for the job.")
@click.option("--log-file", "-o", default=None, help="Log file for the job.")
@click.option("--error-file", "-e", default=None, help="Error file for the job.")
@click.option(
    "--submit/--no-submit",
    default=True,
    help="Submit the job after creating the submit script. Set to False to only create the submit script.",
)
def run(**kwargs):
    """
    The `tatm run` command is used to wrap other tatm commands and run them in a specified compute environment.
    It uses the configuration files and options to determine how to run the command. `tatm run` will use a template
    submit script along with your specified environment to run the command. It will then submit the job to the compute
    environment. If you do not want to submit the job, you can use the `--no-submit` flag. The generated submit script
    will be placed in the current working directory. The submit script will be named `tatm_{command}.submit` where
    `{command}` is the command you are running unless you specify a different name with the `--submit-script` option.

    If the script is not submitted the command will print the submit command to the console.

    The WRAPPED_COMMAND argument
    is the command that will be run. Currently only wrapping ray based commands on slurm is supported.
    """
    config = kwargs.pop("config")
    wrapped_command = kwargs.pop("wrapped_command")
    parse_config_opts(config)
    cfg = load_config()

    options = tatm.compute.run.TatmRunOptions(**kwargs)
    result = tatm.compute.run.run(cfg, options, wrapped_command)
    if not kwargs["submit"]:
        print(" ".join([str(x) for x in result]))
