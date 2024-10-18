from dataclasses import dataclass

from tatm.utils import TatmOptionEnum


class Backend(TatmOptionEnum):
    """Enum class representing the available compute backends for running a command."""

    slurm = "slurm"



@dataclass(kw_only=True)
class Environment:
    """Environment Configuration Class.

    Intended to provide a simple way to configure environment level settings for a compute job. Should hopefully be
    scheduler agnostic, although it is primarily designed for use with Slurm.
    """

    modules: list = (
        None  #: List of modules to load using lmod for the computational environment.
    )
    conda_env: str = None  #: Conda environment to activate for the job.
    singularity_image: str = None  #: Singularity image to use for the job.
    venv: str = (
        None  #: Python virtual environment to activate for the job. Conflicts with conda_env.
    )

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
    """

    nodes: int = 1  #: Number of nodes to use for the job.
    cpus_per_task: int = 1  #: Number of CPUs to use per task.
    gpus_per_node: int = None  #: Number of GPUs to use per node.
    time_limit: str = None  #: Time limit for the job. Expressed in D-HH:MM:SS format.
    memory: str = (
        "0"  #: Memory to allocate for the job. Default is 0, which means all available memory.
    )
    environment: Environment = None  #: Environment configuration for the job.
