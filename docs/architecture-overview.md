# Architecture overview

A short visual summary. For the full UML component, class and sequence diagrams, see
[architecture.md](architecture.md).

## 1. Three model paths, one factory

```mermaid
flowchart TB
  Y["YAML config<br/>variant / model.architecture"] --> F{"build_vlm(cfg)<br/>resolved_architecture()"}
  F -->|hale| H["HaleVLM<br/>SigLIP (frozen) → MLP → Qwen3-8B / R1-7B + LoRA"]
  F -->|halo_moe| M["HaloVLM (scratch)<br/>ViT-MoE → MLP → prefix → 16-layer MoE decoder"]
  F -->|basic| B["BasicVLM (scratch)<br/>OpenCLIP ViT-B/32 → 1 token → TransformerDecoder ×12"]
  H --> T1["generic trainer<br/>optimizer(trainable_parameters)"]
  M --> T2["ScratchVLMTrainer<br/>TensorBoard, gradient stats, videos"]
  B --> T2
```

## 2. Fusion strategies

```mermaid
flowchart LR
  subgraph HaleVLM["HaleVLM: splice at &lt;image&gt;"]
    direction LR
    h1["text"] --> h2["196 SigLIP tokens"] --> h3["text"]
  end
  subgraph HaloVLM["HaloVLM: patch prefix"]
    direction LR
    m1["196 ViT patches"] --> m2["Describe the image … [SEP]"]
  end
  subgraph BasicVLM["BasicVLM: global prefix"]
    direction LR
    b1["1 CLIP token"] --> b2["Describe the image … [SEP]"]
  end
```

## 3. HaloVLM block (encoder and decoder)

```mermaid
flowchart TB
  X["x [B, T, 512]"] --> N1["LayerNorm"] --> A["multi-head attention<br/>(causal in the decoder: 32 heads; ViT: 16 heads)"]
  A --> R1["+ residual"]
  X --> R1
  R1 --> N2["LayerNorm"]
  N2 --> RT["noisy top-4 router over 16 experts"]
  N2 --> S["2 shared SwiGLU experts<br/>512 → 614 → 512"]
  RT --> E["4 routed SwiGLU experts<br/>gate-weighted"]
  N2 --> E
  S --> SUM["Σ"]
  E --> SUM
  SUM --> R2["+ residual"]
  R1 --> R2
```

## 4. Scratch training sequence and targets

```mermaid
flowchart LR
  I["image 224×224"] --> V["ViT-MoE → 196 tokens"] --> P["projector"]
  C["caption"] --> TK["BERT tokenize:<br/>Describe the image + caption + [SEP]"]
  P --> SEQ["[196 patches ; tokens] + positions"]
  TK --> SEQ
  SEQ --> D["16-layer causal MoE decoder"] --> LM["LM head (30,522)"]
  TK --> TG["targets: [-100 × 196 ; ids shifted by 1]"]
  LM --> CE["cross-entropy"]
  TG --> CE
```

## Sizes

| Model | Parameters |
|---|---|
| HaloVLM (scratch, default) | 431.5M total: ViT 108.8M, decoder 288.7M (≈107.6M active/token), embeddings + head 31.3M, positions 2.6M |
| BasicVLM decoder | 37.9M (+ OpenCLIP ViT-B/32) |
| HaleVLM trainable | Qwen3-8B ≈ 83.5M (LoRA 43.6M + projector 39.9M); R1-Distill-7B ≈ 71.6M |
