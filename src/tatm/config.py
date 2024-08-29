import dataclasses
from typing import List, Union, Optional

from omegaconf import MISSING, OmegaConf

from tatm.compute.job import Backend


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
            raise ValueError("Cannot specify both singularity_image and conda_env or venv.")

@dataclasses.dataclass
class TatmConfig:
    backend: Backend = Backend.slurm  #: Backend to use for compute jobs.
    slurm: SlurmConfig = dataclasses.field(
        default_factory=lambda: SlurmConfig(MISSING, MISSING)
    )  #: Slurm specific configuration.
    environment: EnvironmentConfig = dataclasses.field(
        default_factory=lambda: EnvironmentConfig()
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
        config_paths (List[str]): List of paths to load the configuration from.

    Returns:
        TatmConfig: Loaded configuration.
    """
    if not config_paths:
        config_paths = []
    if not overrides:
        overrides = []
    if isinstance(config_paths, str):
        config_paths = [config_paths]
    if isinstance(overrides, str):
        overrides = [overrides]
    cnf = OmegaConf.structured(TatmConfig)

    for path in config_paths:
        cnf = OmegaConf.merge(cnf, OmegaConf.load(path))
    if len(overrides) > 0:
        cnf = OmegaConf.merge(cnf, OmegaConf.from_dotlist(overrides))

    return cnf
