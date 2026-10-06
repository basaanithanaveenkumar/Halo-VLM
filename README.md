# Hale-VLM

Production-quality vision-language modeling library: Hale (HF SigLIP + Qwen/DeepSeek) and scratch (ViT + MoE) stacks, with SmolVLM/VLA data registries.

![Halo VLM](assets/halo_RB.png)

## Resources

| | |
|---|---|
| Paper (arXiv source) | [`paper/main.tex`](paper/main.tex) — build with `make -C paper` |
| Project page | [basaanithanaveenkumar.github.io/Halo-VLM](https://basaanithanaveenkumar.github.io/Halo-VLM/) ([source](project-page/index.html)) |
| Documentation | [`docs/`](docs/index.md) — getting started, configuration, [known issues](docs/known-issues.md) (MkDocs: `docs/mkdocs.yml`) |
| Architecture diagrams | [overview](docs/architecture-overview.md) and [full reference](docs/architecture.md) (Mermaid) |
| Blog | [Three ways to build a vision-language model, in one library](docs/blog/2026-09-28-three-ways-to-build-a-vlm.md) |
| Claude Code skills | [`.claude/skills/`](.claude/skills) — `halo-vlm-dev`, `halo-vlm-scratch`, `halo-vlm-extend`, `hale-publish` |

## Install

```bash
uv sync --extra dev --extra test
```

Scratch/COCO training (optional):

```bash
uv sync --extra dev --extra scratch --extra viz
```

## Library usage

```python
from hale_vlm import load_config, build_vlm

cfg = load_config("configs/base.yaml")
model = build_vlm(cfg)
```

## CLI

```bash
uv run hale-vlm-train configs/qwen3_8b_overfit.yaml
uv run hale-vlm-chat configs/base.yaml --image path/to/image.jpg

# Scratch overfit smoke test
uv run hale-vlm-train configs/halo_moe_overfit.yaml
```

## Vision projectors

`model.vision.projector_type` selects how vision features reach the LLM: `mlp`, `linear`, `qformer` (BLIP-2 style learned queries) or
`gated_cross_attention` (Flamingo style). See [docs/vision_connectors.md](docs/vision_connectors.md).

```bash
uv run hale-vlm-train configs/halo_moe_qformer_overfit.yaml
```

## Package layout

```text
src/hale_vlm/
├── core/          # base config + protocols
├── registry/      # plugin registration
├── config/        # VLM YAML schemas
├── data/          # datasets + registries
├── models/        # Hale + scratch VLMs
├── training/      # trainers + losses
├── inference/     # chat/decode
├── cli/           # entry points
└── utils/         # optim, logging, runtime
configs/           # YAML run configs
examples/          # import-first usage
tests/             # unit + integration
docs/              # mkdocs
```

## Develop

```bash
pre-commit install
uv run pytest tests/ -q
uv run ruff check src tests
```

See [docs/architecture.md](docs/vlm_architecture.md) and [CHANGELOG.md](CHANGELOG.md).

## License

MIT
