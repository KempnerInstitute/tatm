from pathlib import Path

import click


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
        " CLI config options should be in the format `field.subfield=value`."
    ),
    multiple=True,
)
def run(config, wrapped_command):
    files, overrides = parse_config_opts(config)
