# Changelog

## Unreleased

- Added Q-Former (`projector_type: qformer`) and Flamingo-style gated cross-attention (`projector_type: gated_cross_attention`) vision connectors for Hale and scratch VLMs, configurable under `model.vision` (see `docs/vision_connectors.md`)

## 0.2.0

- Unified package under `hale_vlm` with local registry (removed `hale-blocks` dependency)
- Removed legacy `halo_vlm` package and root script shims
- Added `core/`, `registry/`, and `utils/` infrastructure modules
- PyPI package renamed from `halo-vlm` to `hale-vlm`

## 0.1.0

- Initial HaleVLM + scratch VLM merge
