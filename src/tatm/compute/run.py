"""Module holding functions responsible for executing commands wrapped by the "tatm run" CLI command."""

from dataclasses import dataclass

from tatm.config import TatmConfig

from tatm.compute.job import Backend, Environment
from tatm.compute.slurm import SlurmJob, _slurm_create_ray_job


def run(config: TatmConfig, options, command):
    if not command:
        raise ValueError("No command provided to run.")
    if len(command) == 1:
        command = command[0].split()

    if command[0] == "tokenize":
        run_tokenize(config, options, command)
    else:
        raise ValueError(f"Command {command[0]} not supported.")


def run_tokenize(config: TatmConfig, options, command):
    if config.backend == Backend.SLURM:
        job = SlurmJob(
            partition=config.slurm.partition,
            account=config.slurm.account,
            environment=Environment(conda_env="test_env"),
        )
        _slurm_create_ray_job(job, command)
    else:
        raise ValueError(f"Backend {config.backend} not supported.")
    
