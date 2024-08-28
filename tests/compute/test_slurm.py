from tatm.compute.job import Environment
from tatm.compute.slurm import SlurmJob, _slurm_create_ray_job


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
