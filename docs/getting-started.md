# Getting Started

## Install

```bash
git clone https://github.com/basaanithanaveenkumar/Halo-VLM && cd Halo-VLM
uv sync --extra dev --extra test                  # library + tests (Python ≥ 3.12)
uv sync --extra dev --extra scratch --extra viz   # + OpenCLIP, LAVIS COCO, TensorBoard
```

## Test

```bash
uv run pytest tests/ -q          # 29 passed, 2 skipped on CPU
uv run ruff check src tests
```

## Build a model from a config

```python
from hale_vlm import load_config, build_vlm

cfg = load_config("configs/halo_moe_overfit.yaml")   # scratch HaloVLM
model = build_vlm(cfg)
```

## Train

```bash
# scratch HaloVLM (CPU smoke test, then COCO)
uv run hale-vlm-train configs/halo_moe_overfit.yaml
uv run hale-vlm-train configs/halo_moe_coco.yaml

# OpenCLIP baseline
uv run hale-vlm-train configs/basic_vlm_coco.yaml

# pretrained backbone + LoRA (GPU, HF access)
uv run hale-vlm-train configs/qwen3_8b_overfit.yaml
uv run hale-vlm-train configs/smolvlm_vision.yaml
```

## Chat / generate

```bash
uv run hale-vlm-chat configs/base.yaml --image path/to/image.jpg
```

Scratch checkpoints: `hale_vlm.inference.scratch.VLMInference` (greedy, beam search,
streaming).

## Next

- [Architecture overview](architecture-overview.md) and the [full architecture reference](architecture.md)
- [Configuration](configuration.md)
- [Known issues](known-issues.md)
