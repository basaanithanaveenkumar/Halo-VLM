"""Factory and registry for RL training techniques."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.config.sections.rl import RLConfig
from hale_vlm.registry.base import NamedRegistry
from hale_vlm.rl.base import RLTechnique

T = TypeVar("T", bound=type[RLTechnique])

RL_TECHNIQUES = NamedRegistry("rl_technique")

RL_TECHNIQUE_NAMES = (
    "none",
    "dpo",
    "ppo",
    "grpo",
    "reinforce",
    "kto",
    "rloo",
)


def register_rl_technique(name: str) -> Callable[[T], T]:
    def deco(cls: T) -> T:
        RL_TECHNIQUES.add(name, cls)
        cls.technique_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_rl_technique(name: str) -> type[RLTechnique]:
    return RL_TECHNIQUES.get(name)


def build_rl_technique(cfg: RLConfig | VLMRunConfig) -> RLTechnique:
    """Instantiate the configured RL technique (default ``none``)."""
    rl_cfg = cfg.rl if isinstance(cfg, VLMRunConfig) else cfg
    cls = get_rl_technique(rl_cfg.technique)
    return cls(rl_cfg)


def _register_builtin_techniques() -> None:
    from hale_vlm.rl.techniques.dpo import DPORLTechnique
    from hale_vlm.rl.techniques.grpo import GRPORLTechnique
    from hale_vlm.rl.techniques.kto import KTORLTechnique
    from hale_vlm.rl.techniques.none import NoneRLTechnique
    from hale_vlm.rl.techniques.ppo import PPORLTechnique
    from hale_vlm.rl.techniques.reinforce import ReinforceRLTechnique
    from hale_vlm.rl.techniques.rloo import RLOORLTechnique

    for name, cls in (
        ("none", NoneRLTechnique),
        ("dpo", DPORLTechnique),
        ("ppo", PPORLTechnique),
        ("grpo", GRPORLTechnique),
        ("reinforce", ReinforceRLTechnique),
        ("kto", KTORLTechnique),
        ("rloo", RLOORLTechnique),
    ):
        if name not in RL_TECHNIQUES:
            register_rl_technique(name)(cls)


_register_builtin_techniques()
