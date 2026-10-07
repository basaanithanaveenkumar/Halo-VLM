"""Phase-aware VLA dataset registry (pretrain / mid-train / post-train).

Separate from the SmolVLA VLA_DATASETS registry so phase datasets do not change
the count asserted by existing SmolVLA tests.

Usage::

    from hale_vlm.data.vla.phase_registry import (
        build_vla_phase_dataset,
        list_vla_phase_datasets,
        register_vla_phase_dataset,
    )
    from hale_vlm.data.types import VLATrainingPhase

    pretrain_names = list_vla_phase_datasets(phase=VLATrainingPhase.PRETRAIN)
    adapter = build_vla_phase_dataset("bridge-v2")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_vlm.registry.base import NamedRegistry

from hale_vlm.data.vla.adapters.lerobot import VLADataAdapter
from hale_vlm.data.types import VLATrainingPhase

T = TypeVar("T", bound=VLADataAdapter)

# Separate registry — does not pollute the SmolVLA VLA_DATASETS registry.
VLA_PHASE_DATASETS = NamedRegistry("vla_phase_dataset")


def register_vla_phase_dataset(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator that registers a VLADataAdapter in VLA_PHASE_DATASETS."""

    def deco(cls: type[T]) -> type[T]:
        VLA_PHASE_DATASETS.add(name, cls)
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_vla_phase_dataset(name: str) -> type[VLADataAdapter]:
    return VLA_PHASE_DATASETS.get(name)


def build_vla_phase_dataset(name: str, **kwargs) -> VLADataAdapter:
    """Instantiate a VLA phase dataset adapter by registry name."""
    adapter = get_vla_phase_dataset(name)(**kwargs)
    if not adapter.spec.enabled:
        raise ValueError(
            f"VLA phase dataset {name!r} is disabled "
            f"(phase={adapter.spec.phase}): {adapter.spec.description}"
        )
    return adapter


def list_vla_phase_datasets(
    *,
    phase: VLATrainingPhase | None = None,
    enabled_only: bool = True,
) -> list[str]:
    """Return registered VLA phase dataset names, optionally filtered by phase."""
    # Auto-import so registrations run on first call.
    import hale_vlm.data.vla.datasets.phase_datasets  # noqa: F401

    names: list[str] = []
    for name, cls in sorted(VLA_PHASE_DATASETS.items.items()):
        spec = cls.spec
        if enabled_only and not spec.enabled:
            continue
        if phase is not None and spec.phase != phase:
            continue
        names.append(name)
    return names
