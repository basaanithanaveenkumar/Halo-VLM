"""Factory for scratch VLM backends."""

from __future__ import annotations

from hale_vlm.registry import register_model

from hale_vlm.config.run import VLMRunConfig

SCRATCH_VARIANTS = ("basic_vlm_scratch", "halo_vlm_moe")


def build_scratch_vlm(cfg: VLMRunConfig):
    architecture = cfg.model.resolved_architecture(cfg.variant)
    if architecture == "basic":
        from hale_vlm.models.scratch.basic_vlm import BasicVLM

        if cfg.model.scratch.objective != "autoregressive":
            raise ValueError("model.scratch.objective: mtp is implemented for the halo_moe architecture")

        return BasicVLM.from_config(cfg)
    if architecture == "halo_moe":
        from hale_vlm.models.scratch.halo_vlm import HaloVLM

        return HaloVLM.from_config(cfg)
    raise ValueError(f"Unsupported scratch architecture: {architecture}")


def _register_scratch_variants() -> None:
    from hale_vlm.models.scratch.basic_vlm import BasicVLM
    from hale_vlm.models.scratch.halo_vlm import HaloVLM

    @register_model("basic_vlm_scratch")
    class _RegisteredBasicVLM(BasicVLM):
        pass

    @register_model("halo_vlm_moe")
    class _RegisteredHaloVLM(HaloVLM):
        pass


_register_scratch_variants()
