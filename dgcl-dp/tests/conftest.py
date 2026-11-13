import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from dgcl_dp.demo import create_fixture
from dgcl_dp.reproduce import load_task


@pytest.fixture
def config():
    with initialize_config_module(version_base="1.3", config_module="dgcl_dp.configs"):
        cfg = compose(config_name="config", overrides=["experiment=demo"])
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.fixture
def task(tmp_path, config):
    root = tmp_path / "synthetic"
    create_fixture(root)
    return load_task(root, config)
