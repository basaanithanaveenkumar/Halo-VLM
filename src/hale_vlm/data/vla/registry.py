"""VLA / robotics dataset registry (SmolVLA paper)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_vlm.registry.base import NamedRegistry

from hale_vlm.data.vla.adapters.lerobot import VLADataAdapter
from hale_vlm.data.types import VLAStage

T = TypeVar("T", bound=VLADataAdapter)

VLA_DATASETS = NamedRegistry("vla_dataset")


def register_vla_dataset(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        VLA_DATASETS.add(name, cls)
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_vla_dataset(name: str) -> type[VLADataAdapter]:
    return VLA_DATASETS.get(name)


def build_vla_dataset(name: str, **kwargs) -> VLADataAdapter:
    adapter = get_vla_dataset(name)(**kwargs)
    if not adapter.spec.enabled:
        raise ValueError(
            f"VLA dataset {name!r} is disabled in the registry "
            f"(stage={adapter.spec.stage.value}): {adapter.spec.description}"
        )
    return adapter


def list_vla_datasets(
    *,
    stage: VLAStage | None = None,
    enabled_only: bool = True,
) -> list[str]:
    import hale_vlm.data.vla.datasets.builtin  # noqa: F401 — ensure registration

    names: list[str] = []
    for name, cls in sorted(VLA_DATASETS.items.items()):
        spec = cls.spec
        if enabled_only and not spec.enabled:
            continue
        if stage is not None and spec.stage != stage:
            continue
        names.append(name)
    return names
