# Configuration

`VLMRunConfig` is made of strict pydantic sections (unknown keys are errors). YAML files
may `inherits:` a parent.

## Top level

| Key | Meaning |
|---|---|
| `variant` | registered variant: `qwen3_8b_vlm`, `deepseek_r1_qwen_7b_vlm`, `halo_vlm_moe`, `basic_vlm_scratch`, … |
| `device` | e.g. `cpu`, `cuda` |

## `model`

| Key | Default | Meaning |
|---|---|---|
| `architecture` | `auto` | `auto`, `hale`, `basic`, `halo_moe`; `auto` maps from `variant` |
| `max_length` | — | token length |
| `vision.encoder` | `siglip` | `siglip` or `clip` (HaleVLM) |
| `vision.model_id` | `google/siglip-base-patch16-224` | |
| `vision.image_size` | 224 | also used by the scratch ViT |
| `vision.freeze_encoder` | true | |
| `vision.projector_type` / `projector_hidden_dim` / `projector_dropout` | `mlp` / 2·d / 0.0 | |
| `vision.num_image_tokens` | 256 | visual tokens kept (HaleVLM) |
| `llm.backbone` | `qwen3-8b` | `qwen3-8b`, `deepseek-r1-qwen-7b`, `custom` |
| `llm.freeze_llm`, `llm.use_lora` | true, true | |
| `llm.lora_r` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 | |
| `llm.lora_target_modules` | null → q,k,v,o,gate,up,down | |
| `scratch.embed_dim` | 512 | scratch models |
| `scratch.vocab_size`, `scratch.tokenizer_id` | 30522, `bert-base-uncased` | |
| `scratch.patch_size` | 16 | |
| `scratch.vit_num_layers`, `vit_num_heads` | 6, 16 | |
| `scratch.decoder_num_layers`, `decoder_num_heads` | 16, 32 | |
| `scratch.openclip_model`, `openclip_pretrained` | `ViT-B-32`, `laion2b_s34b_b79k` | BasicVLM |
| `scratch.coco_max_length` | 17 | caption tokens |

## `train`

| Key | Default | Meaning |
|---|---|---|
| `backend` | `auto` | `haleblocks` (generic trainer) or `scratch` |
| `batch_size`, `lr`, `steps` / `max_epochs` | per config | |
| `log_tensorboard`, `log_dir` | false, `./runs/hale_vlm` | scratch trainer |
| `scratch_checkpoint_dir` | `checkpoints/scratch` | |

## `data`

| Key | Meaning |
|---|---|
| `source` | `overfit`, `huggingface`, `coco_lavis`, `registry`, `vla_registry`, `mixed_registry` |
| `registry_stage` | `all`, `vision`, `video`, `context` |
| `vla_registry_stage` | `all`, `community`, `simulation`, `real_world` |
| `robotics_vlm_mode` | `off`, `pretraining`, `finetuning`, `instruction_tuning` |
| `max_samples_per_dataset`, `streaming`, `prefetch_workers` | streaming mixer controls |
| `overfit_text`, `n_overfit_copies` | overfit source |
