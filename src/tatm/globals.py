from typing import List

CLI_CONFIG_FILES: List[str] = []
CLI_CONFIG_OVERRIDES: List[str] = []


def set_cli_config_files(files: List[str]):
    global CLI_CONFIG_FILES
    CLI_CONFIG_FILES = files


def set_cli_config_overrides(overrides: List[str]):
    global CLI_CONFIG_OVERRIDES
    CLI_CONFIG_OVERRIDES = overrides


def get_cli_config_files():
    global CLI_CONFIG_FILES
    return CLI_CONFIG_FILES


def get_config_overrides():
    global CLI_CONFIG_OVERRIDES
    return CLI_CONFIG_OVERRIDES
