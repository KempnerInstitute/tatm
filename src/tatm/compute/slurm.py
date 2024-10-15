import logging
import string
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from tatm.compute.job import Job

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SlurmJob(Job):
    """Slurm Job Configuration Class.

    Intended to provide a simple way to configure job level settings for a Slurm compute job.
    """

    partition: str  #: Partition to submit the job to.
    account: str = None  #: Account to charge the job to.
    job_name: str = None  #: Name of the job.
    log_file: str = None  #: Path to the stdout file for the job.
    error_file: str = None  #: Path to the stderr file for the job.
    qos: str = None  #: Quality of Service for the job.
    constraints: Union[str, List[str]] = None
    slurm_bin_dir: str = "/usr/bin/"
    modules: list = None

    def __post_init__(self):
        if isinstance(self.constraints, str):
            self.constraints = [self.constraints]


def _slurm_create_ray_job(job: SlurmJob, command: List[str], job_file_path: str = None):

    if not job_file_path:
        job_file_path = Path.cwd() / f"tatm_{command[0]}_job.submit"

    job_content = _fill_ray_slurm_template(
        job.environment.modules,
        job.environment.conda_env,
        job.environment.singularity_image,
        job.environment.venv,
        command,
    )

    with open(job_file_path, "w") as f:
        f.write(job_content)

    return job_file_path


def _fill_ray_slurm_template(
    modules: List[str],
    conda_env: str,
    singularity_image: str,
    venv: str,
    command: List[str],
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

    if venv:
        options["VENV_ACTIVATE"] = f"source {venv}/bin/activate"
    else:
        options["VENV_ACTIVATE"] = ""

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

    submit_command.extend(["--nodes", str(job.nodes)])
    submit_command.extend(["--cpus-per-task", str(job.cpus_per_task)])
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

    if job.job_name:
        submit_command.extend(["--job-name", job.job_name])

    if job.log_file:
        submit_command.extend(["--output", job.log_file])

    if job.error_file:
        submit_command.extend(["--error", job.error_file])

    submit_command.append(job_file_path)

    return submit_command


def submit_job(
    job: SlurmJob, job_file_path: str, submit: bool = True
) -> Union[str, subprocess.CompletedProcess]:
    """Submit a Slurm job. If submit is False, return the command to submit the job.

    Args:
        job: Instance of SlurmJob containing the job specifications.
        job_file_path (str): Path to the job file to submit.
        submit: Should the job be submitted. Defaults to True. If false, return the command to submit the job for inspection.

    Returns:
        Either the command to submit the job or the result of the submission.
    """
    command = _submit_job_command(job, job_file_path)
    if not submit:
        return command
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        LOGGER.error(
            f"Error submitting job: {e}\nstdout: {e.stdout}\nstderr: {e.stderr}"
        )
        raise ValueError(f"Error submitting job: {e.stderr}")
    except FileNotFoundError as e:
        LOGGER.error(f"Error submitting job: {e.filename} not found.")
        raise ValueError(f"Error submitting job: {e.filename} not found.")

    return result
