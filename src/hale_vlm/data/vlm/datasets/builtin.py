"""Built-in SmolVLM paper dataset registrations."""

from __future__ import annotations

from hale_vlm.data.vlm.adapters.base import VLMDataAdapter
from hale_vlm.data.vlm.adapters.video import VideoDatasetAdapter
from hale_vlm.data.vlm.catalog import (
    SMOLVLM_ALL_DATASETS,
    SMOLVLM_CONTEXT_DATASETS,
    SMOLVLM_REJECTED_DATASETS,
    SMOLVLM_VIDEO_DATASETS,
    SMOLVLM_VISION_DATASETS,
)
from hale_vlm.data.vlm.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory, VisionCategory

# ---------------------------------------------------------------------------
# Vision training stage (§4.1) — Laurençon et al. (2024) mixture + MathWriting
# ---------------------------------------------------------------------------


@register_dataset("the-cauldron")
class TheCauldronAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="the-cauldron",
        hf_path="HuggingFaceM4/the_cauldron",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        category=VisionCategory.OCR_DOCUMENTS.value,
        description="Idefics/SmolVLM vision mixture: OCR, charts, tables, captioning, VQA.",
        paper_reference="Laurençon et al. (2024); SmolVLM §4.1 vision stage",
        image_fields=("images", "image"),
        text_fields=("text", "conversations"),
        conversation_field="conversations",
    )


@register_dataset("docmatix")
class DocmatixAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="docmatix",
        hf_path="HuggingFaceM4/Docmatix",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        category=VisionCategory.OCR_DOCUMENTS.value,
        description="Document understanding and OCR-heavy visual QA.",
        paper_reference="SmolVLM §4.1 vision stage (OCR & Documents ~48%)",
        image_fields=("images", "image"),
        text_fields=("text", "conversations"),
        conversation_field="conversations",
    )


@register_dataset("mathwriting")
class MathWritingAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="mathwriting",
        hf_path="google/mathwriting",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        category=VisionCategory.MATH_HANDWRITING.value,
        description="Handwritten mathematical expression recognition.",
        paper_reference="Gervais et al. (2024); added in SmolVLM §4.1",
        image_fields=("image",),
        text_fields=("text", "latex", "label"),
        conversation_field=None,
    )


@register_dataset("llava-onevision-data")
class LLaVAOneVisionDataAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="llava-onevision-data",
        hf_path="lmms-lab/LLaVA-OneVision-Data",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        category=VisionCategory.CAPTIONING.value,
        description="LLaVA-OneVision image + instruction mix.",
        paper_reference="SmolVLM vision stage + video stage image portion",
        image_fields=("image", "images"),
        text_fields=("text", "caption", "conversations"),
        conversation_field="conversations",
    )


# ---------------------------------------------------------------------------
# Video fine-tuning stage (§4.1)
# ---------------------------------------------------------------------------


@register_dataset("llava-video-178k")
class LLaVAVideo178KAdapter(VideoDatasetAdapter):
    spec = DatasetSpec(
        name="llava-video-178k",
        hf_path="lmms-lab/LLaVA-Video-178K",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.VISUAL_DESCRIPTION.value,
        description="LLaVA-Video-178K captions and open-ended video QA.",
        paper_reference="Zhang et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path", "video_id"),
        text_fields=("text", "caption", "conversations", "question", "answer"),
        conversation_field="conversations",
    )


@register_dataset("videostar")
class VideoStarAdapter(VideoDatasetAdapter):
    spec = DatasetSpec(
        name="videostar",
        hf_path="orrzohar/Video-STaR",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.VISUAL_DESCRIPTION.value,
        description="Video-STaR self-training video instruction tuning.",
        paper_reference="Zohar et al. (2024a); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations", "question", "answer"),
        conversation_field="conversations",
    )


@register_dataset("vript")
class VRiptAdapter(VideoDatasetAdapter):
    spec = DatasetSpec(
        name="vript",
        hf_path="Mutonix/Vript",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.VISUAL_DESCRIPTION.value,
        description="Vript long-form video captioning.",
        paper_reference="Yang et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations"),
        conversation_field="conversations",
    )


@register_dataset("sharegpt4video")
class ShareGPT4VideoAdapter(VideoDatasetAdapter):
    spec = DatasetSpec(
        name="sharegpt4video",
        hf_path="ShareGPT4Video/ShareGPT4Video",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.VISUAL_DESCRIPTION.value,
        description="ShareGPT4Video GPT-4 annotated video instructions.",
        paper_reference="Chen et al. (2023); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations"),
        conversation_field="conversations",
    )


@register_dataset("vista-400k")
class Vista400KAdapter(VideoDatasetAdapter):
    spec = DatasetSpec(
        name="vista-400k",
        hf_path="TIGER-Lab/VISTA-400K",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.TEMPORAL_UNDERSTANDING.value,
        description="VISTA-400K temporal video understanding.",
        paper_reference="Ren et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations", "question", "answer"),
        conversation_field="conversations",
    )


@register_dataset("moviechat")
class MovieChatAdapter(VideoDatasetAdapter):
    spec = DatasetSpec(
        name="moviechat",
        hf_path="Enxin/MovieChat-1K_train",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.NARRATIVE.value,
        description="MovieChat long narrative video comprehension.",
        paper_reference="Song et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations", "question", "answer"),
        conversation_field="conversations",
    )


@register_dataset("finevideo")
class FineVideoAdapter(VideoDatasetAdapter):
    spec = DatasetSpec(
        name="finevideo",
        hf_path="HuggingFaceFV/finevideo",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.NARRATIVE.value,
        description="FineVideo narrative and temporal video understanding.",
        paper_reference="Farré et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path", "mp4"),
        text_fields=("text", "caption", "conversations", "title", "description"),
        conversation_field="conversations",
    )


@register_dataset("m4-instruct-data")
class M4InstructDataAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="m4-instruct-data",
        hf_path="lmms-lab/M4-Instruct-Data",
        modality=Modality.MULTI_IMAGE,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.MULTI_IMAGE.value,
        description="M4 multi-image instruction tuning.",
        paper_reference="Liu et al. (2024a); SmolVLM §4.1",
        image_fields=("images", "image"),
        text_fields=("text", "conversations", "caption"),
        conversation_field="conversations",
    )


@register_dataset("mammoth")
class MammothDataAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="mammoth",
        hf_path="MAmmoTH-VL/MAmmoTH-VL-Instruct-12M",
        modality=Modality.MULTI_IMAGE,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.MULTI_IMAGE.value,
        description="MAmmoTH-VL multi-image instruction corpus.",
        paper_reference="Guo et al. (2024); SmolVLM §4.1",
        image_fields=("images", "image"),
        text_fields=("text", "conversations", "caption"),
        conversation_field="conversations",
    )


@register_dataset("magpie")
class MagpieAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="magpie",
        hf_path="Magpie-Align/Magpie-Pro-300K-Filtered",
        modality=Modality.TEXT,
        stage=TrainingStage.TEXT_SFT,
        category=VideoCategory.TEXT_SFT.value,
        description="Magpie alignment synthesis for the 14% text portion in video stage.",
        paper_reference="Xu et al. (2024); SmolVLM §4.1",
        text_fields=("text", "instruction", "response", "conversations"),
        conversation_field="conversations",
        image_fields=(),
        video_fields=(),
    )


# ---------------------------------------------------------------------------
# Long-context extension (§2.2)
# ---------------------------------------------------------------------------


@register_dataset("dolma-books")
class DolmaBooksAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="dolma-books",
        hf_path="allenai/dolma",
        config_name="books",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="Dolma books subset for 2k→16k context extension.",
        paper_reference="Soldaini et al. (2024); SmolVLM §2.2",
        text_fields=("text", "content"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )


@register_dataset("the-stack")
class TheStackAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="the-stack",
        hf_path="bigcode/the-stack",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="The Stack code corpus for long-context fine-tuning.",
        paper_reference="Kocetkov et al. (2022); SmolVLM §2.2",
        text_fields=("content", "text"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )


@register_dataset("fineweb-edu")
class FineWebEduAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="fineweb-edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="FineWeb-Edu short-context source for context extension.",
        paper_reference="Penedo et al. (2024); SmolVLM §2.2",
        text_fields=("text", "content"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )


@register_dataset("dclm")
class DCLMAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="dclm",
        hf_path="mlfoundations/dclm-baseline-1.0",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="DCLM baseline corpus for context extension.",
        paper_reference="Li et al. (2024a); SmolVLM §2.2",
        text_fields=("text", "content"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )


@register_dataset("smollm2-math")
class SmolLM2MathAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="smollm2-math",
        hf_path="HuggingFaceTB/smollm-corpus",
        config_name="math",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="SmolLM2 math subset used during context extension.",
        paper_reference="Allal et al. (2025); SmolVLM §2.2",
        text_fields=("text", "content"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )


# ---------------------------------------------------------------------------
# Explicitly rejected by SmolVLM (§3.3) — registered but disabled
# ---------------------------------------------------------------------------


@register_dataset("smoltalk")
class SmolTalkRejectedAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="smoltalk",
        hf_path="HuggingFaceTB/smol-smoltalk",
        modality=Modality.TEXT,
        stage=TrainingStage.REJECTED,
        enabled=False,
        description=(
            "REJECTED: reusing LLM-SFT SmolTalk degraded video (-3.7%) and image (-6.5%) "
            "performance. Not used in SmolVLM final mixture."
        ),
        paper_reference="SmolVLM §3.3, Figure 7",
        text_fields=("text", "messages", "conversations"),
        conversation_field="messages",
        image_fields=(),
        video_fields=(),
    )


BUILTIN_DATASET_NAMES: tuple[str, ...] = SMOLVLM_ALL_DATASETS

__all__ = [
    "BUILTIN_DATASET_NAMES",
    "SMOLVLM_ALL_DATASETS",
    "SMOLVLM_CONTEXT_DATASETS",
    "SMOLVLM_REJECTED_DATASETS",
    "SMOLVLM_VIDEO_DATASETS",
    "SMOLVLM_VISION_DATASETS",
]
