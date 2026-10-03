"""3D reference points and camera projection for TPV image cross-attention."""

from __future__ import annotations

import torch


def reference_points_hw(
    tpv_h: int,
    tpv_w: int,
    num_z: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalized 3D anchors for HW (BEV) queries: (num_z, H*W, 3) in [0, 1]."""
    xs = torch.linspace(0.5 / tpv_w, 1.0 - 0.5 / tpv_w, tpv_w, device=device, dtype=dtype)
    ys = torch.linspace(0.5 / tpv_h, 1.0 - 0.5 / tpv_h, tpv_h, device=device, dtype=dtype)
    zs = torch.linspace(0.5 / num_z, 1.0 - 0.5 / num_z, num_z, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    refs = []
    for z in zs:
        refs.append(torch.cat([xy, z.expand(xy.shape[0], 1)], dim=-1))
    return torch.stack(refs, dim=0)


def reference_points_zh(
    tpv_z: int,
    tpv_h: int,
    num_x: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalized 3D anchors for ZH queries: (num_x, Z*H, 3)."""
    zs = torch.linspace(0.5 / tpv_z, 1.0 - 0.5 / tpv_z, tpv_z, device=device, dtype=dtype)
    ys = torch.linspace(0.5 / tpv_h, 1.0 - 0.5 / tpv_h, tpv_h, device=device, dtype=dtype)
    xs = torch.linspace(0.5 / num_x, 1.0 - 0.5 / num_x, num_x, device=device, dtype=dtype)
    zz, yy = torch.meshgrid(zs, ys, indexing="ij")
    zy = torch.stack([zz, yy], dim=-1).reshape(-1, 2)
    refs = []
    for x in xs:
        xyz = torch.stack(
            [x.expand(zy.shape[0]), zy[:, 1], zy[:, 0]],
            dim=-1,
        )
        refs.append(xyz)
    return torch.stack(refs, dim=0)


def reference_points_wz(
    tpv_w: int,
    tpv_z: int,
    num_y: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalized 3D anchors for WZ queries: (num_y, W*Z, 3)."""
    xs = torch.linspace(0.5 / tpv_w, 1.0 - 0.5 / tpv_w, tpv_w, device=device, dtype=dtype)
    zs = torch.linspace(0.5 / tpv_z, 1.0 - 0.5 / tpv_z, tpv_z, device=device, dtype=dtype)
    ys = torch.linspace(0.5 / num_y, 1.0 - 0.5 / num_y, num_y, device=device, dtype=dtype)
    xx, zz = torch.meshgrid(xs, zs, indexing="ij")
    xz = torch.stack([xx, zz], dim=-1).reshape(-1, 2)
    refs = []
    for y in ys:
        xyz = torch.stack(
            [xz[:, 0], y.expand(xz.shape[0]), xz[:, 1]],
            dim=-1,
        )
        refs.append(xyz)
    return torch.stack(refs, dim=0)


def denormalize_points(
    refs: torch.Tensor,
    pc_range: tuple[float, float, float, float, float, float],
) -> torch.Tensor:
    """Map [0, 1] TPV coordinates into the ego / LiDAR volume."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    world = refs.clone()
    world[..., 0] = refs[..., 0] * (x_max - x_min) + x_min
    world[..., 1] = refs[..., 1] * (y_max - y_min) + y_min
    world[..., 2] = refs[..., 2] * (z_max - z_min) + z_min
    return world


def default_lidar2img(
    batch: int,
    num_cams: int,
    image_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Simple surround pinhole cameras for smoke tests when calib is omitted."""
    height, width = image_hw
    fx = fy = float(max(height, width))
    cx, cy = width / 2.0, height / 2.0
    intrinsic = torch.tensor(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    matrices = []
    for cam in range(num_cams):
        yaw = (2.0 * torch.pi * cam) / max(num_cams, 1)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        # Camera sits on a ring looking inward at the ego origin
        extr = torch.eye(4, device=device, dtype=dtype)
        extr[0, 0] = cos_y
        extr[0, 2] = sin_y
        extr[2, 0] = -sin_y
        extr[2, 2] = cos_y
        extr[0, 3] = 2.0 * sin_y
        extr[2, 3] = 2.0 * cos_y
        matrices.append(intrinsic @ extr)
    stacked = torch.stack(matrices, dim=0)
    return stacked.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def project_to_image(
    world_xyz: torch.Tensor,
    lidar2img: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D points to each camera.

    Args:
        world_xyz: (num_anchors, num_queries, 3)
        lidar2img: (B, N, 4, 4)
    Returns:
        pixel_xy: (B, N, num_anchors, num_queries, 2) in [-1, 1]
        valid: (B, N, num_anchors, num_queries) bool
    """
    height, width = image_hw
    num_anchors, num_queries, _ = world_xyz.shape
    batch, num_cams, _, _ = lidar2img.shape
    ones = torch.ones(
        num_anchors,
        num_queries,
        1,
        device=world_xyz.device,
        dtype=world_xyz.dtype,
    )
    homo = torch.cat([world_xyz, ones], dim=-1)
    pts = homo.reshape(-1, 4).T
    cam = torch.matmul(lidar2img.reshape(-1, 4, 4), pts)
    cam = cam.view(batch, num_cams, 4, num_anchors, num_queries)
    depth = cam[:, :, 2]
    u = cam[:, :, 0] / depth.clamp(min=1e-5)
    v = cam[:, :, 1] / depth.clamp(min=1e-5)
    pixel_x = (u / max(width - 1, 1)) * 2.0 - 1.0
    pixel_y = (v / max(height - 1, 1)) * 2.0 - 1.0
    pixel_xy = torch.stack([pixel_x, pixel_y], dim=-1)
    valid = (
        (depth > 1e-5)
        & (pixel_x >= -1.0)
        & (pixel_x <= 1.0)
        & (pixel_y >= -1.0)
        & (pixel_y <= 1.0)
    )
    return pixel_xy, valid
