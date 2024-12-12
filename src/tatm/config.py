import dataclasses
import os
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import MISSING, OmegaConf

from tatm.compute.job import Backend
from tatm.globals import get_cli_config_files, get_config_overrides


@dataclasses.dataclass
class SlurmConfig:
    """Cluster specific configuration for Slurm."""

    partition: str = MISSING  #: Partition to submit jobs to.
    account: Optional[str] = MISSING  #: Account to charge jobs to.
    qos: Optional[str] = None  #: Quality of Service to use for the job.
    slurm_bin_dir: str = "/usr/bin/"  #: Directory containing the Slurm binaries.


@dataclasses.dataclass
class EnvironmentConfig:
    """Environment configuration for compute jobs."""

    modules: Optional[List[str]] = None
    conda_env: Optional[str] = None
    venv: Optional[str] = None
    singularity_image: Optional[str] = None

    def __post_init__(self):
        if self.conda_env is not None and self.venv is not None:
            raise ValueError("Cannot specify both conda_env and venv.")
        if self.singularity_image and (self.conda_env or self.venv):
            raise ValueError(
                "Cannot specify both singularity_image and conda_env or venv."
            )
        if isinstance(self.modules, str):
            self.modules = self.modules.split(",")


@dataclasses.dataclass
class MetadataBackendConfig:
    """Configuration for the metadata backend."""

    type: Optional[str] = None  #: Type of metadata backend to use.
    args: dict = dataclasses.field(
        default_factory=dict
    )  #: Arguments to pass to the metadata backend constructor.


@dataclasses.dataclass
class TatmConfig:
    backend: Backend = Backend.slurm  #: Backend to use for compute jobs.
    slurm: SlurmConfig = dataclasses.field(
        default_factory=lambda: SlurmConfig(MISSING, MISSING)
    )  #: Slurm specific configuration.
    environment: EnvironmentConfig = dataclasses.field(
        default_factory=lambda: EnvironmentConfig()
    )  #: Environment configuration for compute jobs.
    metadata_backend: MetadataBackendConfig = dataclasses.field(
        default_factory=lambda: MetadataBackendConfig()
    )

    def __post_init__(self):
        if not Backend.has_value(self.backend):
            raise ValueError(f"Invalid backend: {self.backend}")

        if self.backend == Backend.SLURM:
            if not self.slurm:
                raise ValueError("Slurm configuration required for SLURM backend.")


def load_config(
    config_paths: Union[List[str], str] = None, overrides: Union[List[str], str] = None
) -> TatmConfig:
    """Load the configuration from the provided paths.

    Args:
        config_paths: List of paths to load the configuration from.
        overrides: List of overrides to apply to the configuration

    Returns:
        TatmConfig: Loaded configuration.
    """
    if config_paths is None:
        config_paths = get_cli_config_files()
    if overrides is None:
        overrides = get_config_overrides()
    if isinstance(config_paths, str):
        config_paths = [config_paths]
    if isinstance(overrides, str):
        overrides = [overrides]
    cnf = load_base_config()

    for path in config_paths:
        cnf = OmegaConf.merge(cnf, OmegaConf.load(path))
    if len(overrides) > 0:
        cnf = OmegaConf.merge(cnf, OmegaConf.from_dotlist(overrides))

    return cnf


def load_base_config() -> TatmConfig:
    """Load the base configuration for TATM. First checks for a configuration file at /etc/tatm/config.yaml, then checks for a configuration file at
    $TATM_BASE_DIR/config/config.yaml and merges them together, with the latter taking precedence. Finally, it checks for a configuration file at
    $TATM_BASE_CONFIG and merges it with the previous configuration, with the latter again taking precedence. If no configuration files are found,
    an empty configuration is returned.

    Returns:
        TatmConfig: Loaded configuration.
    """
    paths = [Path("/etc/tatm/config.yaml")]

    base_dir = os.environ.get("TATM_BASE_DIR")
    if base_dir is not None:
        paths.append(Path(base_dir) / "config" / "config.yaml")

    base_config = os.environ.get("TATM_BASE_CONFIG")
    if base_config is not None:
        paths.append(Path(base_config))

    cnf = OmegaConf.structured(TatmConfig)

    for path in paths:
        if path.exists():
            cnf = OmegaConf.merge(cnf, OmegaConf.load(path))

    return cnf
