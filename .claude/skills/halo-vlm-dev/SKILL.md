---
name: halo-vlm-dev
description: Set up, navigate and test Halo-VLM, the unified `hale_vlm` library with three VLM paths — HaleVLM (SigLIP + LoRA on Qwen3-8B / DeepSeek-R1-Distill-Qwen-7B), scratch HaloVLM (ViT + MoE decoder) and BasicVLM (OpenCLIP + TransformerDecoder) — plus SmolVLM/SmolVLA data registries. Use when starting work here, before changing library code, or when picking which model path a config uses.
---

# Halo-VLM development

The package is `hale_vlm` (v0.2.0). It is **self-contained**: config, registry and
runtime utilities are vendored under `hale_vlm/core`, `hale_vlm/registry` and
`hale_vlm/utils`, so no HaleBlocks install is needed.

## Environment

```bash
uv sync --extra dev --extra test                      # library + tests (Python ≥ 3.12)
uv sync --extra dev --extra scratch --extra viz       # + OpenCLIP, LAVIS COCO, TensorBoard
pre-commit install
```

## Which model does a config build?

`build_vlm(cfg)` (`models/vlm.py`) resolves `cfg.model.resolved_architecture(cfg.variant)`:

| `variant` | architecture | Class | Training backend |
|---|---|---|---|
| `qwen3_8b_vlm`, `deepseek_r1_qwen_7b_vlm` | `hale` | `HaleVLM` (`models/vlm.py`) | generic trainer (LoRA + projector only) |
| `halo_vlm_moe` (or any variant with `model.architecture: halo_moe`, e.g. `halo_moe_overfit.yaml`) | `halo_moe` | `HaloVLM` (`models/scratch/halo_vlm.py`) | `train.backend: scratch` → `ScratchVLMTrainer` |
| `basic_vlm_scratch` | `basic` | `BasicVLM` (`models/scratch/basic_vlm.py`) | scratch |

`model.architecture` overrides the variant mapping. `configs/smolvlm_*` and `smolvla_*`
select **data**, not a model.

## Code map

| Path | What |
|---|---|
| `core/config/` | typed sections (strict pydantic), YAML `inherits:` merge, loader |
| `registry/` | `register_model`, `register_loss`, `register_trainer`, dataset registries, plugin bootstrap |
| `config/` | `VLMRunConfig`: `model.vision`, `model.llm`, `model.scratch`, `model.architecture`, VLM data/train sections |
| `vision/`, `llm/` | SigLIP/CLIP tower, MLP projector, HF LLM + PEFT LoRA |
| `models/scratch/components/` | `vit.py`, `transformer.py`, `moe.py` (DeepSeekMoE), `image_proj.py`, `lm_head.py`, OpenCLIP/CLIP encoders |
| `data/` | SmolVLM/SmolVLA catalogs and adapters, streaming mixer, `coco_lavis.py`, `scratch_encoding.py`, `multimodal.py` |
| `training/` | `losses.py` (HF loss for Hale, shifted CE for scratch), `trainer.py`, `scratch_trainer.py`, `schedule.py`, `distributed.py` |
| `inference/scratch.py` | `VLMInference`: greedy, beam search, token streaming |
| `docs/` | MkDocs site (`docs/mkdocs.yml`); `architecture.md` has the full UML/Mermaid set |

## Tests (verified: 29 passed, 2 skipped on CPU)

```bash
uv run pytest tests/ -q
uv run ruff check src tests
```

`tests/helpers/tiny_models.py` provides tiny stand-ins. Never download 8B weights in tests.

## Sizes (measured, default `ScratchConfig`)

Scratch HaloVLM has 431.5M parameters: ViT 108.8M (6 layers, MoE 16 routed + 2 shared,
top-4), decoder 288.7M (16 layers, 32 heads, same MoE; ≈107.6M active per token), token
embedding and LM head 15.6M each (BERT vocab 30,522), positions 2.6M. The BasicVLM
decoder is 37.9M (12 × `nn.TransformerDecoderLayer`, d = 512, 64 heads, FFN 1024) plus
OpenCLIP ViT-B/32.

## Known issues

- **HaleVLM image merge overwrites text** (same as Hale-VLM): one `<image>` placeholder
  is replaced by `num_image_tokens` positions, dropping the following text. See
  `docs/known-issues.md`.
- **HaleVLM loss covers the prompt**: `labels = input_ids` except padding.
- Scratch `HaloVLM.forward` ignores `attention_mask` (causal mask only), so inputs must be
  right-padded.
- `format_scratch_caption` concatenates without a space: `"Describe the image" + caption`.
- `docs/vlm_architecture.md` is a byte-identical copy of `docs/architecture.md`. Edit
  both or remove one.
