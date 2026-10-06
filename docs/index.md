# Halo-VLM

Vision-language modeling library with three model paths behind one config schema and
factory:

- **HaleVLM**: frozen SigLIP + MLP projector + Qwen3-8B / DeepSeek-R1-Distill-Qwen-7B with LoRA
- **HaloVLM**: from-scratch ViT + 16-layer causal decoder with DeepSeek-style MoE (431.5M parameters)
- **BasicVLM**: OpenCLIP global token + `nn.TransformerDecoder` baseline

| Page | Contents |
|---|---|
| [Getting started](getting-started.md) | install, test, train, chat |
| [Architecture overview](architecture-overview.md) | four Mermaid diagrams and model sizes |
| [Architecture reference](architecture.md) | full UML / class / sequence diagrams |
| [Configuration](configuration.md) | every config section |
| [Known issues](known-issues.md) | fix before benchmarking |
| [Blog](blog/README.md) | long-form posts |
| [Notes](halo_notes.md) | research notes |

Also: the [paper](../paper/main.tex) and the [project page](../project-page/index.html).
