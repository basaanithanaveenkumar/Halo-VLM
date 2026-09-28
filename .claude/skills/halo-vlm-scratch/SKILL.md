---
name: halo-vlm-scratch
description: Train, visualise and run inference with the from-scratch Halo-VLM models (HaloVLM ViT+MoE, BasicVLM OpenCLIP) on COCO captions via LAVIS or overfit data. Use when asked to train the scratch model, debug its loss, generate captions (greedy/beam/streaming), or read its TensorBoard/video outputs.
---

# Scratch VLMs

## Configs

| Config | Model | Data | Notes |
|---|---|---|---|
| `configs/halo_moe_overfit.yaml` | HaloVLM | overfit text ×8 | CPU smoke test, 1 epoch |
| `configs/halo_moe_coco.yaml` | HaloVLM | COCO captions (LAVIS) | bs 8, lr 3e-4, 10 epochs, TensorBoard in `runs/halo_moe_coco` |
| `configs/basic_vlm_coco.yaml` | BasicVLM | COCO captions | bs 4, lr 1e-4, 2 epochs |

```bash
uv sync --extra dev --extra scratch --extra viz
uv run hale-vlm-train configs/halo_moe_overfit.yaml
uv run hale-vlm-train configs/halo_moe_coco.yaml
tensorboard --logdir runs
```

All scratch configs use `bert-base-uncased` (vocab 30,522) and `train.backend: scratch`.

## Sequence and loss

```
[ 196 image patch tokens | "Describe the image<caption>" [SEP] pad… ]
targets: [ -100 × 196    | input_ids shifted left by one, pad → -100 ]
```

`encode_scratch_text` builds labels that are already shifted, so `_scratch_vlm_loss`
compares `logits[:, t]` with `targets[:, t]` directly. Don't shift again. BasicVLM uses
one global image token (`num_image_tokens = 1`) instead of 196.

## Inference

`hale_vlm/inference/scratch.py::VLMInference(model_path, device)` provides
`generate_greedy`, `generate_beam_search` and `generate_token_stream` (yields tokens as
they are produced). Prompts start with `"Describe the image"`.

## Debugging

| Symptom | Check |
|---|---|
| Loss stuck around ln(30522) ≈ 10.3 | labels all `-100` (`coco_max_length` too small) or the learning rate is too high for MoE |
| Captions repeat one word | router collapse: log per-expert token counts; the MoE has no load-balancing loss |
| OOM | 196 patches + text × 16 decoder layers × 32 heads; lower `batch_size` or `decoder_num_layers` in `model.scratch` |
| Gradient logs | `ScratchVLMTrainer.log_gradients` / `check_gradients` write to TensorBoard |
