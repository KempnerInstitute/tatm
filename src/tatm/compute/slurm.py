import string
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from tatm.compute.job import Job


@dataclass(kw_only=True)
class SlurmJob(Job):
    """Slurm Job Configuration Class.

    Intended to provide a simple way to configure job level settings for a Slurm compute job.

    Args:
        partition (str): Partition to submit jobs to.
        account (str): Account to charge jobs to.
        slurm_bin_dir (str): Directory containing the Slurm binaries.
    """

    partition: str
    account: str = None
    qos: str = None
    constraints: Union[str,List[str]] = None
    slurm_bin_dir: str = "/usr/bin/"
    modules: list = None

    def __post_init__(self):
        if isinstance(self.constraints, str):
            self.constraints = [self.constraints]


def _slurm_create_ray_job(job: SlurmJob, command: List[str], job_file_path: str = None):

    if not job_file_path:
        job_file_path = Path.cwd() / f"tatm_{command[0]}_job.submit"

    job_content = _fill_ray_slurm_template(
        job.modules,
        job.environment.conda_env,
        job.environment.singularity_image,
        command,
    )

    with open(job_file_path, "w") as f:
        f.write(job_content)


def _fill_ray_slurm_template(
    modules: List[str], conda_env: str, singularity_image: str, command: List[str]
):
    with open(Path(__file__).parent / "templates" / "slurm" / "ray.submit") as f:
        job_template = string.Template(f.read())

    options = {}

    if modules:
        options["MODULES"] = "module load " + " ".join(modules)
    else:
        options["MODULES"] = ""

    if conda_env:
        options["CONDA_ACTIVATE"] = f"conda activate {conda_env}"
    else:
        options["CONDA_ACTIVATE"] = ""

    if singularity_image:
        options["SINGULARITY_WRAP"] = f"singularity exec {singularity_image}"
    else:
        options["SINGULARITY_WRAP"] = ""

    options["TATM_CMD"] = "tatm " + " ".join(command)

    job_content = job_template.safe_substitute(**options)

    return job_content

def _submit_job_command(job: SlurmJob, job_file_path: str):
    sbatch_path = str(Path(job.slurm_bin_dir) / "sbatch")

    submit_command = [sbatch_path]

    submit_command.extend(["--nodes", job.nodes])
    submit_command.extend(["--cpus-per-task", job.cpus_per_task])
    if job.gpus_per_node:
        submit_command.extend(["--gres", f"gpu:{job.gpus_per_node}"])
    submit_command.extend(["--mem", job.memory])
    if job.time_limit:
        submit_command.extend(["--time", job.time_limit])

    if job.qos:
        submit_command.extend(["--qos", job.qos])
    submit_command.extend(["--partition", job.partition])
    if job.account:
        submit_command.extend(["--account", job.account])

    if job.constraints:
        submit_command.extend(["--constraint", ",".join(job.constraints)])
    
    


    submit_command.append(job_file_path)

    return submit_command