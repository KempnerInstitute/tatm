"""Module holding functions responsible for executing commands wrapped by the "tatm run" CLI command."""

from dataclasses import dataclass

from tatm.compute.job import Backend, Environment
from tatm.compute.slurm import SlurmJob, _slurm_create_ray_job, submit_job
from tatm.config import TatmConfig


@dataclass
class TatmRunOptions:
    """Options for the "tatm run" command."""

    nodes: int = None
    cpus_per_task: int = None
    submit_script: str = None
    time_limit: str = None
    memory: str = None
    gpus_per_node: int = None
    constraints: str = None
    submit: bool = (
        True  #: Submit the job after creating the submit script. Set to False to only create the submit script. (default: True)
    )


def run(config: TatmConfig, options: TatmRunOptions, command):
    if not command:
        raise ValueError("No command provided to run.")
    if len(command) == 1:
        command = command[0].split()

    if command[0] == "tokenize":
        return run_tokenize(config, options, command)
    else:
        raise ValueError(f"Command {command[0]} not supported.")


def run_tokenize(config: TatmConfig, options: TatmRunOptions, command):
    if config.backend == Backend.slurm:
        env = Environment(
            conda_env=config.environment.conda_env,
            modules=config.environment.modules,
            singularity_image=config.environment.singularity_image,
            venv=config.environment.venv,
        )
        print(env)
        job = SlurmJob(
            partition=config.slurm.partition,
            account=config.slurm.account,
            memory=options.memory,
            time_limit=options.time_limit,
            nodes=options.nodes,
            cpus_per_task=options.cpus_per_task,
            gpus_per_node=options.gpus_per_node,
            constraints=options.constraints,
            environment=env,
        )
        submit_script = _slurm_create_ray_job(job, command, options.submit_script)

        if options.submit:
            return submit_job(job, submit_script, submit=True)
        else:
            return submit_job(job, submit_script, submit=False)
    else:
        raise ValueError(f"Backend {config.backend} not supported.")
