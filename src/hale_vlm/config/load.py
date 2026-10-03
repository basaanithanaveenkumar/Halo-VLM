"""Load and inherit YAML into a validated config schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel

from hale_vlm.config.merge import deep_merge
from hale_vlm.registry import get_config_schema


def load_yaml_tree(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    logger.debug("loading config from {}", path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if "inherits" in raw:
        parent = (path.parent / raw["inherits"]).resolve()
        parent_raw = yaml.safe_load(parent.read_text()) or {}
        if "inherits" in parent_raw:
            grand = (parent.parent / parent_raw["inherits"]).resolve()
            grand_raw = yaml.safe_load(grand.read_text()) or {}
            parent_raw = deep_merge(grand_raw, parent_raw)
        raw = deep_merge(parent_raw, raw)
    return raw


def validate_config[T: BaseModel](raw: dict[str, Any], schema: type[T]) -> T:
    return schema.model_validate(raw)


def load_registered_config(path: str | Path, schema: str = "vlm") -> BaseModel:
    path = Path(path).resolve()
    model_cls = get_config_schema(schema)
    cfg = validate_config(load_yaml_tree(path), model_cls)
    if hasattr(cfg, "experiment") and getattr(cfg.experiment, "source_yaml", None) is not None:
        cfg.experiment.source_yaml = str(path)
    logger.info(
        "loaded config schema={} variant={} from {}",
        schema,
        getattr(cfg, "variant", "?"),
        path,
    )
    return cfg
