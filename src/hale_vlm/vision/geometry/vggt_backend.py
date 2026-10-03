"""Optional VGGT-Ω backend placeholder."""

from __future__ import annotations

from hale_vlm.config.sections.gwm import GWMConfig
from hale_vlm.vision.geometry.encoder import MultiViewGeometryEncoder


class VGGTGeometryEncoder(MultiViewGeometryEncoder):
    """Placeholder for a frozen VGGT-Ω checkpoint integration.

    Until official weights/loaders are wired, this falls back to the local encoder
    with ``geometry_backend='local'`` semantics.
    """

    def __init__(self, cfg: GWMConfig) -> None:
        super().__init__(cfg)
