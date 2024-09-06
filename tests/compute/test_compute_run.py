from tatm.compute.run import TatmRunOptions, _add_default_options, run
from tatm.config import load_config


def test_run_tokenize(tmp_path):
    output_path = tmp_path / "run.submit"
    config = load_config("tests/compute/configs/test_config.yaml")
    options = TatmRunOptions(
        nodes=1,
        cpus_per_task=1,
        submit_script=str(output_path),
        time_limit="1-00:00:00",
        memory="10G",
        submit=False,
    )
    command = ["tokenize", "--num-workers", "10", "test_data"]
    submit_command = run(config, options, command)
    expected_command = [
        "/usr/bin/sbatch",
        "--nodes",
        "1",
        "--cpus-per-task",
        "1",
        "--mem",
        "10G",
        "--time",
        "1-00:00:00",
        "--partition",
        "debug",
        "--account",
        "kempner_dev",
        "--job-name",
        "tatm_tokenize",
        "--output",
        "tatm_tokenize.out",
        str(tmp_path / "run.submit"),
    ]
    for i in range(len(expected_command)):
        assert submit_command[i] == expected_command[i]

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


def test_set_option_defaults():
    options = TatmRunOptions(
        nodes=2,
        cpus_per_task=None,
        submit_script=None,
        time_limit=None,
        memory="25G",
        gpus_per_node=None,
        constraints=None,
        submit=True,
    )
    defaults = TatmRunOptions(
        nodes=1,
        cpus_per_task=40,
        submit_script=None,
        time_limit="1-00:00:00",
        memory="40G",
        gpus_per_node=None,
        constraints=None,
        submit=True,
    )
    _add_default_options(options, defaults)
    assert options.nodes == 2
    assert options.cpus_per_task == 40
    assert options.submit_script is None
    assert options.time_limit == "1-00:00:00"
    assert options.memory == "25G"
    assert options.gpus_per_node is None
    assert options.constraints is None
    assert options.submit is True
