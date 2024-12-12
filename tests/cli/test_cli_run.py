import pytest
from click.testing import CliRunner

from tatm.cli.run import parse_config_opts, run
from tatm.globals import set_cli_config_files, set_cli_config_overrides


@pytest.fixture
def temp_global_config():
    set_cli_config_files([])
    set_cli_config_overrides([])
    yield
    set_cli_config_files([])
    set_cli_config_overrides([])


def test_parse_config_opts(temp_global_config):
    opts = ["config1", "config2", "field.subfield=value"]
    files, overrides = parse_config_opts(opts, validate=False)
    assert files == ["config1", "config2"]
    assert overrides == ["field.subfield=value"]


def test_run_tokenize(tmp_path, temp_global_config):
    runner = CliRunner()
    expected_return = "/usr/bin/sbatch --nodes 1 --cpus-per-task 40 --mem 40G --time 1-00:00:00 --partition debug --account kempner_dev --job-name tatm_tokenize --output tatm_tokenize.out run.submit"
    with runner.isolated_filesystem(temp_dir=tmp_path):
        args = [
            "-N",
            "1",
            "-c",
            "40",
            "--conf",
            "slurm.partition=debug",
            "--conf",
            "slurm.account=kempner_dev",
            "--submit-script",
            "run.submit",
            "--no-submit",
            "tokenize",
            "test_data",
        ]
        result = runner.invoke(run, args)
        print(result.output)
        assert result.exit_code == 0
        output = result.output.strip()
        assert output == expected_return
