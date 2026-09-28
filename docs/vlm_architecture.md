# VLM Architecture Reference

This document compares every vision-language model (VLM) path in Halo-VLM and provides UML structural and class diagrams in [Mermaid](https://mermaid.js.org/) format.

---

## Table of Contents

1. [At a Glance](#1-at-a-glance)
2. [Model Comparison](#2-model-comparison)
3. [Package Structure (UML Component)](#3-package-structure-uml-component)
4. [HaleVLM — Class Diagram](#4-halevlm--class-diagram)
5. [Legacy Halo-VLM — Class Diagrams](#5-legacy-halo-vlm--class-diagrams)
6. [Data Pipeline — Class Diagram](#6-data-pipeline--class-diagram)
7. [Forward-Pass Sequence Diagrams](#7-forward-pass-sequence-diagrams)
8. [Config → Target Mapping](#8-config--target-mapping)
9. [Entry Points](#9-entry-points)
10. [Unified Package (Post-Merge)](#10-unified-package-post-merge)

---

## 1. At a Glance

Halo-VLM ships **three model architectures** and **two data registries** that reuse the same `HaleVLM` model:

| Name | Package | Type | Role |
|------|---------|------|------|
| **HaleVLM** | `hale_vlm` | Production model | Pretrained SigLIP/CLIP + projector + Qwen3 / DeepSeek-R1 LLM |
| **BasicVLM** | `hale_vlm.models.scratch` | Scratch model | OpenCLIP global feature + PyTorch `TransformerDecoder` |
| **HaloVLM** | `hale_vlm.models.scratch` | Scratch model | Custom ViT + MoE decoder transformer |
| **SmolVLM registry** | `hale_vlm.data` | Data catalog | 19 HF multimodal datasets (vision / video / context stages) |
| **SmolVLA registry** | `hale_vlm.data` | Data catalog | Robotics demonstration datasets (community / sim / real) |
| **Robotics VLM adapter** | `hale_vlm.data` | Data converter | Maps `VLASample` → `VLMSample` with mode-specific prompts |

> **Important:** Configs named `smolvlm_*` and `smolvla_*` select **training data**, not a separate model class. Scratch models use configs such as `halo_moe_coco.yaml`, `basic_vlm_coco.yaml`, or `halo_moe_overfit.yaml`.

---

## 10. Unified Package (Post-Merge)

All VLM code paths now route through **`hale_vlm`**:

```mermaid
flowchart TB
    CLI["hale-vlm-train / hale-vlm-chat"]
    CFG["VLMRunConfig"]
    FACTORY["build_vlm()"]
    DATA["MultimodalDataModule"]

    subgraph Backends
        Hale["HaleVLM\ntoken_replace"]
        Basic["BasicVLM\nprefix_concat"]
        Halo["HaloVLM\nprefix_concat"]
    end

    subgraph Trainers
        HaleTrainer["VLMTrainer\nhaleblocks"]
        ScratchTrainer["ScratchVLMTrainer"]
    end

    CLI --> CFG
    CFG --> FACTORY
    CFG --> DATA
    FACTORY --> Hale
    FACTORY --> Basic
    FACTORY --> Halo
    DATA -->|"pixel_values batches"| HaleTrainer
    DATA -->|"images batches"| ScratchTrainer
    Hale --> HaleTrainer
    Basic --> ScratchTrainer
    Halo --> ScratchTrainer
```

| Config | Model | Data source | Trainer |
|--------|-------|-------------|---------|
| `base.yaml` | HaleVLM | overfit / registry | `VLMTrainer` |
| `halo_moe_coco.yaml` | HaloVLM | `coco_lavis` | `ScratchVLMTrainer` |
| `halo_moe_overfit.yaml` | HaloVLM | overfit | `ScratchVLMTrainer` |
| `smolvlm_*.yaml` | HaleVLM | SmolVLM registry | `VLMTrainer` |

All training and inference run through **`hale-vlm-train`** and **`hale-vlm-chat`** with local plugin registries under `hale_vlm/registry/`.

---

## 2. Model Comparison

### 2.1 Architecture

| Dimension | HaleVLM | BasicVLM | HaloVLM |
|-----------|---------|----------|---------|
| **Philosophy** | Assemble pretrained HF components | Train small stack from scratch | Custom ViT + MoE decoder research stack |
| **Vision encoder** | SigLIP / CLIP / AutoModel (HF) | OpenCLIP ViT-B-32 | Custom `VisTransformer` (6 layers) |
| **Language model** | Qwen3-8B or DeepSeek-R1-Qwen-7B | 12-layer PyTorch `TransformerDecoder` | 16-layer `DecoderTransformer` with MoE |
| **Projector** | `VisionProjector` (linear or 2-layer MLP) | `ImageProjector` (3-layer MLP) | `ImageProjector` (3-layer MLP) |
| **Image fusion** | In-place `<image>` token replacement | Prefix: 1 global image token | Prefix: 196 patch tokens (14×14) |
| **Positional encoding** | Inside pretrained LLM | Sinusoidal (fixed) | Learned `nn.Embedding` |
| **Loss** | HF causal LM (`labels` → `outputs.loss`) | Manual cross-entropy; image positions masked | Same as BasicVLM |
| **Fine-tuning** | LoRA on LLM; optional vision unfreeze | Full-model AdamW | Full-model AdamW |
| **Training infra** | `hale-vlm-train`, YAML configs, `VLMTrainer` | `halo_vlm.train.BasicVLMTrainer` | Same trainer path as BasicVLM |
| **Inference** | `hale-vlm-chat` → `model.generate()` | `halo_vlm.inference.VLMInference` | Same inference path as BasicVLM |

### 2.2 Image Fusion Strategies

```mermaid
flowchart LR
    subgraph HaleVLM["HaleVLM — token replacement"]
        A1["Prompt: User: &lt;image&gt;\nDescribe…"] --> A2["Embed text tokens"]
        A2 --> A3["Replace &lt;image&gt; span\nwith N projected patches"]
        A3 --> A4["Pretrained causal LM"]
    end

    subgraph Legacy["BasicVLM / HaloVLM — prefix concat"]
        B1["Image encoder"] --> B2["Projector"]
        B3["Token embedding"] --> B4["concat(image, text)"]
        B2 --> B4
        B4 --> B5["Custom decoder"]
    end
```

### 2.3 Registered Variants (HaleVLM only)

| Registry name | LLM backbone | Config file |
|---------------|--------------|-------------|
| `qwen3_8b_vlm` | `Qwen/Qwen3-8B` | `configs/base.yaml`, `configs/qwen3_8b_overfit.yaml` |
| `deepseek_r1_qwen_7b_vlm` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `configs/deepseek_r1_qwen_7b.yaml` |

Both variants share the same `HaleVLM` class; only the LLM preset and `reasoning_mode` flag differ.

---

## 3. Package Structure (UML Component)

High-level module layout and dependencies.

```mermaid
flowchart TB
    subgraph CLI["CLI Layer"]
        train_cli["hale_vlm.cli.train"]
        chat_cli["hale_vlm.cli.chat"]
    end

    subgraph Config["Configuration"]
        run_cfg["VLMRunConfig"]
        vision_cfg["VisionConfig"]
        llm_cfg["LLMConfig"]
        data_cfg["DataConfig"]
    end

    subgraph Model["Model Layer — hale_vlm"]
        hale_vlm["HaleVLM"]
        vision_tower["VisionTower"]
        projector["VisionProjector"]
        llm_backbone["LLMBackbone"]
    end

    subgraph Data["Data Layer — hale_vlm"]
        multimodal["MultimodalDataModule"]
        registry["SmolVLM Registry"]
        vla_registry["SmolVLA Registry"]
        robotics["robotics_vlm adapter"]
    end

    subgraph Training["Training Layer"]
        vlm_trainer["VLMTrainer"]
        losses["VLM losses"]
        bootstrap["bootstrap.register_vlm_plugins"]
    end

    subgraph Legacy["Legacy — halo_vlm"]
        basic_vlm["BasicVLM"]
        halo_vlm_cls["HaloVLM"]
        legacy_train["BasicVLMTrainer"]
        legacy_infer["VLMInference"]
    end

    subgraph External["External Dependencies"]
        hf["HuggingFace Transformers"]
        hale_core["hale_core registry / Trainer"]
        openclip["OpenCLIP"]
    end

    train_cli --> run_cfg
    train_cli --> bootstrap
    chat_cli --> hale_vlm
    bootstrap --> hale_vlm
    bootstrap --> vlm_trainer

    run_cfg --> hale_vlm
    hale_vlm --> vision_tower
    hale_vlm --> projector
    hale_vlm --> llm_backbone

    vision_tower --> hf
    llm_backbone --> hf
    vlm_trainer --> hale_core
    vlm_trainer --> hale_vlm

    multimodal --> registry
    multimodal --> vla_registry
    vla_registry --> robotics
    multimodal --> hale_vlm

    basic_vlm --> openclip
    halo_vlm_cls --> basic_vlm
    legacy_train --> basic_vlm
    legacy_infer --> basic_vlm
```

### Repository layout

```mermaid
flowchart LR
    subgraph src_hale["src/hale_vlm/"]
        models["models/vlm.py"]
        vision["vision/"]
        llm["llm/"]
        data["data/"]
        training["training/"]
        cli["cli/"]
        config["config/"]
    end

    subgraph src_halo["src/halo_vlm/"]
        legacy_models["models/"]
        legacy_train["train.py"]
        legacy_infer["inference.py"]
    end

    subgraph configs_dir["configs/"]
        base["base.yaml"]
        smolvlm["smolvlm_*.yaml"]
        smolvla["smolvla_*.yaml"]
        robotics_cfg["vlm_with_robotics_*.yaml"]
    end

    configs_dir --> models
    models --> vision
    models --> llm
    data --> models
    training --> models
```

---

## 4. HaleVLM — Class Diagram

Production VLM assembled from pretrained vision and language components.

```mermaid
classDiagram
    direction TB

    class HaleVLM {
        +VLMRunConfig cfg
        +int image_token_id
        +trainable_parameters() Iterator
        +encode_images(pixel_values) Tensor
        +merge_image_embeddings(input_ids, text_embeds, image_embeds) Tuple
        +forward(input_ids, attention_mask, pixel_values, labels) CausalLMOutput
        +from_config(cfg) HaleVLM
    }

    class VisionTower {
        +VisionConfig cfg
        +str encoder_type
        +Module model
        +int hidden_size
        +forward(pixel_values) Tensor
    }

    class VisionProjector {
        +Module net
        +forward(vision_features) Tensor
    }

    class LLMBackbone {
        +LLMConfig cfg
        +PreTrainedTokenizer tokenizer
        +Module model
        +int hidden_size
        +int vocab_size
        +embed_tokens(input_ids) Tensor
        +forward(inputs_embeds, input_ids, attention_mask, labels) CausalLMOutput
    }

    class VLMRunConfig {
        +str variant
        +ModelConfig model
        +TrainConfig train
        +DataConfig data
    }

    class VisionConfig {
        +str encoder
        +str model_id
        +bool freeze_encoder
        +int num_image_tokens
        +str projector_type
    }

    class LLMConfig {
        +str backbone
        +str model_id
        +bool freeze_llm
        +bool use_lora
        +bool reasoning_mode
        +str image_token
    }

    class VLMTrainer {
        +fit()
    }

    class MultimodalDataModule {
        +train_dataloader() DataLoader
        +val_dataloader() DataLoader
    }

    HaleVLM *-- VisionTower : vision
    HaleVLM *-- VisionProjector : projector
    HaleVLM *-- LLMBackbone : llm
    HaleVLM ..> VLMRunConfig : configured by

    VisionTower ..> VisionConfig
    VisionProjector ..> VisionConfig
    LLMBackbone ..> LLMConfig
    VLMRunConfig *-- VisionConfig
    VLMRunConfig *-- LLMConfig

    VLMTrainer --> HaleVLM : optimizes trainable_parameters
    MultimodalDataModule --> HaleVLM : batches

    note for HaleVLM "Registered as qwen3_8b_vlm\nand deepseek_r1_qwen_7b_vlm"
    note for VisionTower "SiglipVisionModel | CLIPVisionModel | AutoModel"
    note for LLMBackbone "AutoModelForCausalLM + optional PEFT LoRA"
```

### HaleVLM factory and registry

```mermaid
classDiagram
    direction LR

    class HaleVLM
    class RegisteredQwen3 {
        <<registered>>
        qwen3_8b_vlm
    }
    class RegisteredDeepSeek {
        <<registered>>
        deepseek_r1_qwen_7b_vlm
    }

    class build_vlm {
        <<function>>
        +build_vlm(cfg) HaleVLM
    }

    HaleVLM <|-- RegisteredQwen3
    HaleVLM <|-- RegisteredDeepSeek
    build_vlm ..> HaleVLM : from_config
```

---

## 5. Legacy Halo-VLM — Class Diagrams

### 5.1 BasicVLM

OpenCLIP produces a **single global image token** prepended to the text sequence.

```mermaid
classDiagram
    direction TB

    class BasicVLM {
        +Embedding token_embeds
        +SinusoidalPositionalEmbedding positional_embeds
        +OpenCLIPEncoder vision_encoder
        +ImageProjector image_projector
        +TransformerDecoder transformer
        +LMHead lm_head
        +forward(images, input_ids, attention_mask) Tensor
    }

    class OpenCLIPEncoder {
        +str model_name
        +bool freeze
        +forward(images) Tensor
    }

    class ImageProjector {
        +Sequential proj
        +forward(img_features) Tensor
    }

    class SinusoidalPositionalEmbedding {
        +forward(x) Tensor
    }

    class LMHead {
        +LayerNorm norm
        +Linear head
        +forward(hidden) Tensor
    }

    class TransformerDecoder {
        <<PyTorch>>
        12 layers, 64 heads
    }

    BasicVLM *-- OpenCLIPEncoder
    BasicVLM *-- ImageProjector
    BasicVLM *-- SinusoidalPositionalEmbedding
    BasicVLM *-- TransformerDecoder
    BasicVLM *-- LMHead

    note for BasicVLM "Output: [B, 1 + seq_len, vocab_size]"
```

### 5.2 HaloVLM

Custom ViT emits **196 patch tokens**; decoder blocks use **Mixture-of-Experts** FFN.

```mermaid
classDiagram
    direction TB

    class HaloVLM {
        +VisTransformer vis_enc
        +DecoderTransformer decoder_transformer
        +Embedding token_emb
        +Embedding pos_embed
        +LayerNorm layer_norm
        +ImageProjector image_projector
        +LMHead lm_head
        +forward(images, input_ids, attention_mask) Tensor
    }

    class VisTransformer {
        +PatchEmb patch_emb
        +Parameter pos_embed
        +ModuleList transformer
        +forward(x) Tensor
    }

    class PatchEmb {
        +Conv2d proj
        +forward(x) Tensor
    }

    class DecoderTransformer {
        +ModuleList layers
        +forward(x) Tensor
    }

    class TransformerBlock {
        +MultiHeadAttn attn
        +DeepseekMoE ffn
        +forward(x) Tensor
    }

    class DeepseekMoE {
        +ModuleList experts
        +Linear gate
        +forward(x) Tensor
    }

    HaloVLM *-- VisTransformer
    HaloVLM *-- DecoderTransformer
    HaloVLM *-- ImageProjector
    HaloVLM *-- LMHead

    VisTransformer *-- PatchEmb
    VisTransformer *-- TransformerBlock
    DecoderTransformer *-- TransformerBlock
    TransformerBlock *-- DeepseekMoE

    note for HaloVLM "Output: [B, 196 + seq_len, vocab_size]"
    note for VisTransformer "6 layers, 16 heads, patch 16×16"
    note for DecoderTransformer "16 layers, 32 heads, MoE FFN"
```

### 5.3 Shared legacy components

```mermaid
classDiagram
    direction LR

    class ImageProjector
    class LMHead
    class BasicVLM
    class HaloVLM

    BasicVLM --> ImageProjector
    BasicVLM --> LMHead
    HaloVLM --> ImageProjector
    HaloVLM --> LMHead

    note for ImageProjector "3-layer MLP with LayerNorm + GELU\nvision_dim → llm_dim"
    note for LMHead "LayerNorm + Linear → vocab logits"
```

---

## 6. Data Pipeline — Class Diagram

SmolVLM, SmolVLA, and robotics modes are **data paths** into the same `HaleVLM` training loop.

```mermaid
classDiagram
    direction TB

    class VLMSample {
        +str dataset
        +Modality modality
        +str text
        +list images
        +dict metadata
    }

    class VLASample {
        +str task
        +list images
        +Tensor action
        +Tensor state
        +Embodiment embodiment
    }

    class MultimodalSample {
        +Tensor pixel_values
        +Tensor input_ids
        +Tensor attention_mask
        +Tensor labels
    }

    class MultimodalDataset {
        +list records
        +Tokenizer tokenizer
        +getitem(idx) MultimodalSample
    }

    class RegistryStreamingDataset {
        +iter_registry_vlm_samples()
    }

    class VLADataAdapter {
        +adapt(row) VLASample
    }

    class robotics_vlm {
        <<module>>
        +vla_sample_to_vlm(sample, mode) VLMSample
        +format_robotics_instruction(task, mode) str
    }

    class RoboticsVLMMode {
        <<enumeration>>
        PRETRAINING
        FINETUNING
        INSTRUCTION_TUNING
    }

    class SmolVLMCatalog {
        <<registry>>
        vision stage datasets
        video stage datasets
        context stage datasets
    }

    class SmolVLACatalog {
        <<registry>>
        community datasets
        simulation datasets
        real_world datasets
    }

    SmolVLMCatalog ..> VLMSample : yields
    SmolVLACatalog ..> VLASample : yields
    VLADataAdapter --> VLASample
    VLASample --> robotics_vlm
    robotics_vlm --> VLMSample
    VLMSample --> MultimodalDataset
    RegistryStreamingDataset --> MultimodalDataset
    MultimodalDataset --> MultimodalSample
    MultimodalSample --> HaleVLM : training batch

    robotics_vlm ..> RoboticsVLMMode
```

### Robotics prompt modes

| Mode | Prompt template | Typical use |
|------|-----------------|-------------|
| `pretraining` | `Robot task: {task}` | `vlm_with_robotics_pretrain.yaml` |
| `finetuning` | `Execute the following manipulation task: {task}` | `vlm_with_robotics_finetune.yaml` |
| `instruction_tuning` | `Instruction: {task}\nDescribe the robot action…` | Custom / eval prompts |

---

## 7. Forward-Pass Sequence Diagrams

### 7.1 HaleVLM

```mermaid
sequenceDiagram
    autonumber
    participant DL as DataLoader
    participant V as VisionTower
    participant P as VisionProjector
    participant L as LLMBackbone
    participant M as HaleVLM.merge_image_embeddings

    DL->>L: input_ids, attention_mask, labels
    DL->>V: pixel_values [B,3,H,W]
    V->>V: SigLIP / CLIP forward
    V-->>P: patch features [B, N, D_v]
    P-->>M: projected [B, N, D_llm]
    L->>L: embed_tokens(input_ids)
    L-->>M: text_embeds [B, seq, D_llm]
    M->>M: replace image-token span
    M-->>L: inputs_embeds, attention_mask
    L->>L: causal LM forward (+ LoRA)
    L-->>DL: loss, logits
```

### 7.2 BasicVLM

```mermaid
sequenceDiagram
    autonumber
    participant M as BasicVLM
    participant VE as OpenCLIPEncoder
    participant IP as ImageProjector
    participant TE as token_embeds
    participant TR as TransformerDecoder
    participant LH as LMHead

    M->>VE: images [B,3,224,224]
    VE-->>IP: global feature [B,1,D]
    IP-->>M: img_proj [B,1,D]
    M->>TE: input_ids
    TE-->>M: text_embeds [B,seq,D]
    M->>M: concat + sinusoidal positions
    M->>TR: combined_embeds, causal mask
    TR-->>LH: hidden [B,1+seq,D]
    LH-->>M: logits [B,1+seq,vocab]
```

### 7.3 HaloVLM

```mermaid
sequenceDiagram
    autonumber
    participant M as HaloVLM
    participant VT as VisTransformer
    participant IP as ImageProjector
    participant DT as DecoderTransformer
    participant LH as LMHead

    M->>VT: images [B,3,224,224]
    VT-->>IP: patch tokens [B,196,D]
    IP-->>M: img_proj [B,196,D]
    M->>M: token_emb + learned pos_embed
    M->>M: concat(image, text)
    M->>DT: combined_embeds (MoE blocks)
    DT-->>M: hidden states
    M->>LH: layer_norm → logits
    LH-->>M: [B,196+seq,vocab]
```

---

## 8. Config → Target Mapping

All configs live in `configs/` and drive **HaleVLM** training unless noted.

```mermaid
flowchart TB
    subgraph ModelConfigs["Model variant configs"]
        base["base.yaml\nqwen3_8b_vlm"]
        overfit["qwen3_8b_overfit.yaml"]
        deepseek["deepseek_r1_qwen_7b.yaml\ndeepseek_r1_qwen_7b_vlm"]
    end

    subgraph SmolVLMConfigs["SmolVLM data configs → HaleVLM + registry"]
        all["smolvlm_all.yaml\nstage: all (19 datasets)"]
        vision["smolvlm_vision.yaml\nstage: vision"]
        video["smolvlm_video.yaml\nstage: video"]
        context["smolvlm_context.yaml\nstage: context"]
        mix["registry_mix.yaml\nstage: all"]
    end

    subgraph SmolVLAConfigs["SmolVLA data configs → HaleVLM + vla_registry"]
        vla_all["smolvla_all.yaml"]
        community["smolvla_community.yaml"]
        simulation["smolvla_simulation.yaml"]
        real["smolvla_real_world.yaml"]
    end

    subgraph RoboticsConfigs["Robotics + VLM hybrid configs"]
        pretrain["vlm_with_robotics_pretrain.yaml\nmixed_registry + pretraining mode"]
        finetune["vlm_with_robotics_finetune.yaml\nvision registry + finetuning mode"]
    end

    HaleVLM["HaleVLM model"]

    ModelConfigs --> HaleVLM
    SmolVLMConfigs --> HaleVLM
    SmolVLAConfigs --> HaleVLM
    RoboticsConfigs --> HaleVLM
```

| Config | Model variant | Data source | Notes |
|--------|---------------|-------------|-------|
| `base.yaml` | `qwen3_8b_vlm` | `overfit` | Default local smoke config |
| `qwen3_8b_overfit.yaml` | `qwen3_8b_vlm` | `overfit` | Alias of base |
| `deepseek_r1_qwen_7b.yaml` | `deepseek_r1_qwen_7b_vlm` | inherits base data | `reasoning_mode: true` |
| `smolvlm_all.yaml` | inherits base | `registry` / `all` | Full SmolVLM mix |
| `smolvlm_vision.yaml` | inherits base | `registry` / `vision` | Image-heavy datasets |
| `smolvlm_video.yaml` | inherits base | `registry` / `video` | Video datasets |
| `smolvlm_context.yaml` | inherits base | `registry` / `context` | Long-context text |
| `registry_mix.yaml` | inherits base | `registry` / `all` | Same as `smolvlm_all` |
| `smolvla_all.yaml` | inherits base | `vla_registry` / `all` | All robotics stages |
| `smolvla_community.yaml` | inherits base | `vla_registry` / `community` | Community SO-100 HF sets |
| `smolvla_simulation.yaml` | inherits base | `vla_registry` / `simulation` | LIBERO, MetaWorld |
| `smolvla_real_world.yaml` | inherits base | `vla_registry` / `real_world` | Real robot SO-100/101 |
| `vlm_with_robotics_pretrain.yaml` | inherits base | `mixed_registry` | SmolVLM all + robotics pretrain prompts |
| `vlm_with_robotics_finetune.yaml` | inherits base | `registry` + robotics | Vision stage + finetune prompts |

---

## 9. Entry Points

| Stack | Train | Chat / Infer |
|-------|-------|--------------|
| **hale_vlm** | `uv run hale-vlm-train configs/<name>.yaml` | `uv run hale-vlm-chat configs/base.yaml --image <path>` |
| **halo_vlm** | `PYTHONPATH=src uv run python -m halo_vlm.train` | `PYTHONPATH=src uv run python -m halo_vlm.inference --model <ckpt> --image <path>` |

Plugin registration (`hale_vlm.bootstrap.register_vlm_plugins`) wires:

- **Config:** `VLMRunConfig`
- **Models:** `qwen3_8b_vlm`, `deepseek_r1_qwen_7b_vlm`
- **Trainer:** `vlm` → `VLMTrainer` (optimizer sees only `trainable_parameters()`)
- **Loss:** per-variant handlers delegating to HF `outputs.loss`

---

## Related Docs

- [`dataloader.md`](./dataloader.md) — legacy COCO dataloader notes
- [`halo_notes.md`](./halo_notes.md) — research reading list and fusion taxonomy
