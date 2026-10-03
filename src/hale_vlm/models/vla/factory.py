"""VLA model factory and variant registration."""

from __future__ import annotations

from hale_vlm.registry import register_model

from hale_vlm.config.run import VLMRunConfig

VLA_VARIANTS = ("gwm_vla",)


def build_vla_model(cfg: VLMRunConfig):
    architecture = cfg.model.resolved_architecture(cfg.variant)
    if architecture == "gwm_vla":
        from hale_vlm.models.vla.gwm_vla import GWMVLA

        return GWMVLA.from_config(cfg)
    raise ValueError(f"Unsupported VLA architecture: {architecture}")


def _register_vla_variants() -> None:
    from hale_vlm.models.vla.gwm_vla import GWMVLA

    @register_model("gwm_vla")
    class _RegisteredGWMVLA(GWMVLA):
        pass


_register_vla_variants()
