import pytest

from tatm.config import EnvironmentConfig

def test_environment_config():
    env = EnvironmentConfig()
    assert env.singularity_image is None
    assert env.conda_env is None
    assert env.venv is None
    assert env.modules is None
    with pytest.raises(ValueError):
        EnvironmentConfig(conda_env="test_env", venv="test_venv")
    with pytest.raises(ValueError):
        EnvironmentConfig(singularity_image="test_image", conda_env="test_env")
    with pytest.raises(ValueError):
        EnvironmentConfig(singularity_image="test_image", venv="test_venv")