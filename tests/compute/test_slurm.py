from tatm.compute.job import Environment
from tatm.compute.slurm import SlurmJob, _slurm_create_ray_job, _submit_job_command


def test_slurm_create_ray_job(tmp_path):
    env = Environment(conda_env="test_env")
    job = SlurmJob(
        partition="partition", account="account", environment=env, modules=["python"]
    )
    output_path = tmp_path / "ray.submit"
    command = ["tokenize", "--num-workers", "10", "test_data"]
    _slurm_create_ray_job(job, command, output_path)
    with open(output_path) as f:
        content = f.readlines()
    with open("tests/compute/templates/slurm/ray.submit") as f:
        expected_content = f.readlines()

    for i in range(len(content)):
        if content[i] != expected_content[i]:
            print(f"Line {i} does not match")
            print(f"Expected: {expected_content[i]}")
            print(f"Got: {content[i]}")
            assert content[i] == expected_content[i]


def test_slurm_create_ray_singularity_job(tmp_path):
    env = Environment(singularity_image="test.sif")
    job = SlurmJob(partition="partition", account="account", environment=env)
    output_path = tmp_path / "ray.submit"
    command = ["tokenize", "--num-workers", "10", "test_data"]
    _slurm_create_ray_job(job, command, output_path)
    with open(output_path) as f:
        content = f.readlines()
    with open("tests/compute/templates/slurm/ray_singularity.submit") as f:
        expected_content = f.readlines()

    for i in range(len(content)):
        if content[i] != expected_content[i]:
            print(f"Line {i} does not match")
            print(f"Expected: {expected_content[i]}")
            print(f"Got: {content[i]}")
            assert content[i] == expected_content[i]

def test_submit_command():
    job = SlurmJob(partition="partition", account="account",  nodes=2, cpus_per_task=4, time_limit="1-00:00:00", memory="10G", gpus_per_node=1, constraints="h100", qos="high")

    job_file_path = "test_job_file"

    command = _submit_job_command(job, job_file_path)
    expected_command = [
        "/usr/bin/sbatch",
        "--nodes",
        2,
        "--cpus-per-task",
        4,
        "--gres",
        "gpu:1",
        "--mem",
        "10G",
        "--time",
        "1-00:00:00",
        "--qos",
        "high",
        "--partition",
        "partition",
        "--account",
        "account",
        "--constraint",
        "h100",
        job_file_path,
    ]
    for i in range(len(command)):
        if command[i] != expected_command[i]:
            print(f"argument {i} does not match")
            print(f"Expected: {expected_command[i]}")
            print(f"Got: {command[i]}")
            assert command[i] == expected_command[i]

