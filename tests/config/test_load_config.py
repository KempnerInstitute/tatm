from tatm.config import load_config


def test_load_config():

    config = load_config("tests/config/examples/base.yaml")
    assert config.backend == "slurm"
    assert config.slurm.partition == "debug"

    config = load_config(
        ["tests/config/examples/base.yaml", "tests/config/examples/base2.yaml"]
    )
    assert config.backend == "slurm"
    assert config.slurm.partition == "test"
    assert config.slurm.account == "kempner_dev"
    assert config.slurm.slurm_bin_dir == "/usr/local/bin/"

    config = load_config(
        ["tests/config/examples/base.yaml", "tests/config/examples/base2.yaml"],
        "slurm.partition=debug2",
    )
    assert config.backend == "slurm"
    assert config.slurm.partition == "debug2"
    assert config.slurm.slurm_bin_dir == "/usr/local/bin/"

    config = load_config(
        "tests/config/examples/base.yaml",
        ["slurm.partition=debug2", "slurm.account=kempner_dev2"],
    )
    assert config.backend == "slurm"
    assert config.slurm.partition == "debug2"
    assert config.slurm.account == "kempner_dev2"
