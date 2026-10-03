from hale_vlm.config.load import load_registered_config
from hale_vlm.config.run import RunConfig, VLMRunConfig
from hale_vlm.config.sections.common import StrictModel
from hale_vlm.config.sections.model import VLMModelConfig
from hale_vlm.registry import register_config

register_config("vlm")(VLMRunConfig)


def load_vlm_config(path: str) -> VLMRunConfig:
    cfg = load_registered_config(path, schema="vlm")
    assert isinstance(cfg, VLMRunConfig)
    return cfg


load_config = load_vlm_config

__all__ = [
    "RunConfig",
    "StrictModel",
    "VLMModelConfig",
    "VLMRunConfig",
    "load_config",
    "load_registered_config",
    "load_vlm_config",
]
