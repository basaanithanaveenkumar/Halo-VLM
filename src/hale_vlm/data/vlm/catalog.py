"""SmolVLM paper dataset catalog and stage presets."""

from __future__ import annotations

from hale_vlm.data.types import TrainingStage, VideoCategory, VisionCategory

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
