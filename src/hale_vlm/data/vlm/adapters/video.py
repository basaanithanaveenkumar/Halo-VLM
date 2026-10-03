"""Video-aware adapter helpers."""

from __future__ import annotations

from typing import Any

from PIL import Image

from hale_vlm.data.vlm.adapters.base import VLMDataAdapter, _as_image


def load_video_frames(
    source: Any,
    *,
    max_frames: int = 8,
    image_size: int | None = None,
) -> list[Image.Image]:
    """Decode a small, evenly spaced set of frames without keeping the full video on disk."""
    if source is None:
        return []

    if isinstance(source, list):
        frames = [_as_image(item) for item in source]
        return [frame for frame in frames if frame is not None][:max_frames]

    path = str(source)
    try:
        from torchvision.io import read_video
    except ImportError:
        return []

    video, _, _ = read_video(path, pts_unit="sec")
    if video.numel() == 0:
        return []

    total = int(video.shape[0])
    if total <= max_frames:
        indices = list(range(total))
    else:
        step = max(total // max_frames, 1)
        indices = list(range(0, total, step))[:max_frames]

    frames: list[Image.Image] = []
    for idx in indices:
        frame = video[idx].permute(2, 0, 1).numpy()
        image = Image.fromarray(frame)
        if image_size is not None:
            image = image.resize((image_size, image_size))
        frames.append(image.convert("RGB"))
    return frames


class VideoDatasetAdapter(VLMDataAdapter):
    max_video_frames: int = 8
    image_size: int | None = None

    def _extract_video_frames(self, row: dict[str, Any]) -> list[Image.Image]:
        for key in self.spec.video_fields:
            if key in row and row[key] is not None:
                frames = load_video_frames(
                    row[key],
                    max_frames=self.max_video_frames,
                    image_size=self.image_size,
                )
                if frames:
                    return frames
        return []
