"""Public registry API."""

from hale_vlm.registry.base import NamedRegistry
from hale_vlm.registry.plugins import (
    build_logger,
    get_config_schema,
    get_dataset,
    get_loss,
    get_model,
    get_trainer,
    get_variant,
    list_datasets,
    register_config,
    register_dataset,
    register_logger,
    register_loss,
    register_model,
    register_trainer,
)
from hale_vlm.rl.factory import get_rl_technique, register_rl_technique

__all__ = [
    "NamedRegistry",
    "build_logger",
    "get_config_schema",
    "get_dataset",
    "get_loss",
    "get_model",
    "get_rl_technique",
    "get_trainer",
    "get_variant",
    "list_datasets",
    "register_config",
    "register_dataset",
    "register_logger",
    "register_loss",
    "register_model",
    "register_rl_technique",
    "register_trainer",
]
