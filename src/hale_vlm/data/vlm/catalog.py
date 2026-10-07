"""SmolVLM paper dataset catalog, stage presets, and training-phase presets."""

from __future__ import annotations

from hale_vlm.data.types import TrainingPhase, TrainingStage, VideoCategory, VisionCategory

# Vision stage (§4.1) — Laurençon et al. (2024) mixture + MathWriting
SMOLVLM_VISION_DATASETS: tuple[str, ...] = (
    "the-cauldron",
    "docmatix",
    "mathwriting",
    "llava-onevision-data",
)

# Video fine-tuning stage (§4.1)
SMOLVLM_VIDEO_DATASETS: tuple[str, ...] = (
    "llava-video-178k",
    "videostar",
    "vript",
    "sharegpt4video",
    "vista-400k",
    "moviechat",
    "finevideo",
    "m4-instruct-data",
    "mammoth",
    "magpie",
)

# Long-context extension (§2.2)
SMOLVLM_CONTEXT_DATASETS: tuple[str, ...] = (
    "dolma-books",
    "the-stack",
    "fineweb-edu",
    "dclm",
    "smollm2-math",
)

# Full SmolVLM training mixture (vision + video + context + text SFT)
SMOLVLM_ALL_DATASETS: tuple[str, ...] = (
    *SMOLVLM_VISION_DATASETS,
    *SMOLVLM_VIDEO_DATASETS,
    *SMOLVLM_CONTEXT_DATASETS,
)

# Explicitly excluded by the paper (§3.3) — registered for documentation only
SMOLVLM_REJECTED_DATASETS: tuple[str, ...] = ("smoltalk",)

STAGE_PRESETS: dict[TrainingStage, tuple[str, ...]] = {
    TrainingStage.VISION: SMOLVLM_VISION_DATASETS,
    TrainingStage.VIDEO: SMOLVLM_VIDEO_DATASETS,
    TrainingStage.CONTEXT: SMOLVLM_CONTEXT_DATASETS,
    TrainingStage.TEXT_SFT: ("magpie",),
    TrainingStage.REJECTED: SMOLVLM_REJECTED_DATASETS,
}

VISION_CATEGORY_NOTES: dict[VisionCategory, str] = {
    VisionCategory.OCR_DOCUMENTS: "48% of vision mixture",
    VisionCategory.CAPTIONING: "14% of vision mixture",
    VisionCategory.CHART_UNDERSTANDING: "12% of vision mixture",
    VisionCategory.REASONING_LOGIC: "9% visual + 79% text portion",
    VisionCategory.TABLE_UNDERSTANDING: "9% of vision mixture",
    VisionCategory.VISUAL_QA: "8% of vision mixture (incl. 2% multi-image)",
    VisionCategory.GENERAL_KNOWLEDGE: "21% of vision-stage text portion",
    VisionCategory.MATH_HANDWRITING: "Added for handwritten math OCR (Gervais et al., 2024)",
}

VIDEO_CATEGORY_NOTES: dict[VideoCategory, str] = {
    VideoCategory.VISUAL_DESCRIPTION: "LLaVA-Video-178K, Video-STaR, Vript, ShareGPT4Video",
    VideoCategory.TEMPORAL_UNDERSTANDING: "VISTA-400K",
    VideoCategory.NARRATIVE: "MovieChat, FineVideo",
    VideoCategory.MULTI_IMAGE: "M4-Instruct, Mammoth",
    VideoCategory.TEXT_SFT: "Magpie (Xu et al., 2024); 14% text in video stage",
}

# ---------------------------------------------------------------------------
# Training-phase dataset groups (orthogonal to the SmolVLM stage taxonomy)
# These live in a separate phase registry — they do NOT affect SMOLVLM_* counts.
# ---------------------------------------------------------------------------

# Phase 1 — PRETRAIN: large-scale weakly-supervised image-text alignment
PRETRAIN_VLM_DATASETS: tuple[str, ...] = (
    "laion-aesthetics-v2-5plus",  # LAION-Aesthetics filtered (≥5.0 quality)
    "cc3m",                        # Conceptual Captions 3M
    "cc12m",                       # Conceptual Captions 12M
    "datacomp-1b",                 # DataComp-1B CommonPool
    "wit",                         # Wikipedia Image Text (Google)
    "redcaps",                     # RedCaps: 12M captions from Reddit
)

# Phase 2 — MID_TRAIN: connector pre-training / curated domain adaptation
MID_TRAIN_VLM_DATASETS: tuple[str, ...] = (
    "llava-pretrain-558k",   # LLaVA image-caption 558K (BLIP-generated)
    "sharegpt4v-pt",         # ShareGPT4V 1.2M high-quality captions
    "blip-laion-cc-sbu-558k", # BLIP pretraining mix (LAION+CC+SBU)
    "recap-datacomp-1b",     # DataComp-1B re-captioned with LLaVA
    "allava-vflan",          # ALLaVA VFLAN curated instruction mix
)

# Phase 3 — POST_TRAIN: instruction-following SFT and task fine-tuning
POST_TRAIN_VLM_DATASETS: tuple[str, ...] = (
    "llava-instruct-665k",   # LLaVA-1.5 SFT 665K
    "textvqa",               # TextVQA (text in natural images)
    "scienceqa",             # ScienceQA multi-modal QA
    "chartqa",               # ChartQA chart understanding
    "infographics-vqa",      # InfographicsVQA document VQA
    "seed-bench",            # SEED-Bench spatial & temporal reasoning
    "llava-plus",            # LLaVA-Plus skill-augmented instruction data
)

VLM_PHASE_PRESETS: dict[TrainingPhase, tuple[str, ...]] = {
    TrainingPhase.PRETRAIN: PRETRAIN_VLM_DATASETS,
    TrainingPhase.MID_TRAIN: MID_TRAIN_VLM_DATASETS,
    TrainingPhase.POST_TRAIN: POST_TRAIN_VLM_DATASETS,
}

PHASE_NOTES: dict[TrainingPhase, str] = {
    TrainingPhase.PRETRAIN: (
        "Large-scale noisy image-text pairs for initial vision-language alignment. "
        "Trains the visual encoder + projector from scratch or continues pretrained LLM."
    ),
    TrainingPhase.MID_TRAIN: (
        "Curated higher-quality image-text and caption data. "
        "Bridges pretrain features to instruction-following capability."
    ),
    TrainingPhase.POST_TRAIN: (
        "Task-specific instruction-following SFT. "
        "Trains the full VLM on multi-turn QA, OCR, chart and reasoning benchmarks."
    ),
}
