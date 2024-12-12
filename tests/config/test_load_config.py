import pytest

from tatm.config import load_config
from tatm.globals import set_cli_config_files, set_cli_config_overrides


@pytest.fixture
def temp_global_config():
    set_cli_config_files([])
    set_cli_config_overrides([])
    yield
    set_cli_config_files([])
    set_cli_config_overrides([])


class TestConfigLoading:

    def test_load_base_config(self):
        config = load_config("tests/config/examples/base.yaml")
        assert config.backend == "slurm"
        assert config.slurm.partition == "debug"

    def test_merge_configs(self):
        config = load_config(
            ["tests/config/examples/base.yaml", "tests/config/examples/base2.yaml"]
        )
        assert config.backend == "slurm"
        assert config.slurm.partition == "test"
        assert config.slurm.account == "kempner_dev"
        assert config.slurm.slurm_bin_dir == "/usr/local/bin/"

    def test_override_config(self):
        config = load_config(
            ["tests/config/examples/base.yaml", "tests/config/examples/base2.yaml"],
            "slurm.partition=debug2",
        )
        assert config.backend == "slurm"
        assert config.slurm.partition == "debug2"
        assert config.slurm.slurm_bin_dir == "/usr/local/bin/"

    def test_override_config_multiple(self):
        config = load_config(
            "tests/config/examples/base.yaml",
            ["slurm.partition=debug2", "slurm.account=kempner_dev2"],
        )
        assert config.backend == "slurm"
        assert config.slurm.partition == "debug2"
        assert config.slurm.account == "kempner_dev2"

    def test_load_metadata_backend_config(self):
        config = load_config("tests/config/examples/metadata_backend_only.yaml")
        assert config.metadata_backend.type == "json"
        assert config.metadata_backend.args == {"metadata_store_path": "metadata.json"}

    def test_load_env_var_config(self, monkeypatch):
        monkeypatch.setenv("TATM_BASE_DIR", "tests/config/examples")
        config = load_config()
        assert config.backend == "slurm"
        assert config.slurm.partition == "debug"
        assert config.slurm.account == "kempner_dev"

    def test_load_env_var_config_override(self, monkeypatch):
        monkeypatch.setenv("TATM_BASE_DIR", "tests/config/examples")
        monkeypatch.setenv("TATM_BASE_CONFIG", "tests/config/examples/base2.yaml")
        config = load_config()
        assert config.backend == "slurm"
        assert config.slurm.partition == "test"
        assert config.slurm.account == "kempner_dev"
        assert config.slurm.slurm_bin_dir == "/usr/local/bin/"

    def test_load_env_var_config_override_cli(self, monkeypatch, temp_global_config):
        monkeypatch.setenv("TATM_BASE_DIR", "tests/config/examples")
        monkeypatch.setenv(
            "TATM_BASE_CONFIG", "tests/config/examples/metadata_backend_only.yaml"
        )
        set_cli_config_files(["tests/config/examples/base2.yaml"])
        config = load_config()
        assert config.backend == "slurm"
        assert config.slurm.partition == "test"
        assert config.slurm.account == "kempner_dev"
        assert config.slurm.slurm_bin_dir == "/usr/local/bin/"
        assert config.metadata_backend.type == "json"
        assert config.metadata_backend.args == {"metadata_store_path": "metadata.json"}
