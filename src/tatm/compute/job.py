from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    """Enum class representing the available compute backends for running a command."""

    slurm = "slurm"

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


@dataclass(kw_only=True)
class Environment:
    """Environment Configuration Class.

    Intended to provide a simple way to configure environment level settings for a compute job. Should hopefully be
    scheduler agnostic, although it is primarily designed for use with Slurm.

    Args:
        modules (list): List of modules to load.
        conda_env (str): Conda environment to activate.
        singularity_image (str): Singularity image to use. Will wrap the job in a singularity call if provided. The wrapped
            command will be `singularity exec <singularity_image> <command>`.
    """

    modules: list = None
    conda_env: str = None
    singularity_image: str = None

    def __post_init__(self):
        if self.modules is None:
            self.modules = []
        if self.conda_env is None:
            self.conda_env = ""
        if self.singularity_image is None:
            self.singularity_image = ""


@dataclass(kw_only=True)
class Job:
    """Job Level Configuration Class.

    Intended to provide a simple way to configure job level settings for a compute job. Should hopefully be
    scheduler agnostic, although it is primarily designed for use with Slurm.

    Args:
        nodes (int): Number of nodes to use for the job.
        cpus_per_task (int): Number of CPUs to use per task.
        gpus_per_node (int): Number of GPUs to use per node.
        memory (str): Memory to allocate for the job. Default is 0, which means all available memory.
    """

    nodes: int = 1  #: Number of nodes to use for the job.
    cpus_per_task: int = 1  #: Number of CPUs to use per task.
    gpus_per_node: int = None  #: Number of GPUs to use per node.
    time_limit: str = None #: Time limit for the job. Expressed in D-HH:MM:SS format.
    memory: str = (
        "0"  #: Memory to allocate for the job. Default is 0, which means all available memory.
    )
    environment: Environment = None  #: Environment configuration for the job.
