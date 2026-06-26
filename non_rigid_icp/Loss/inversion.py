"""Triangle inversion / area-collapse barrier.

A self-intersection is often preceded by a local fold: a triangle's normal
flips relative to its rest orientation, or its area collapses toward zero. This
barrier penalizes both, which keeps the surface locally embedded (no foldover)
and complements the pairwise collision / sheet barriers. It is purely local
(per-triangle), so it is cheap and always evaluated over all faces.
"""

import torch


def triangleAreaNormals(vertices: torch.Tensor, faces: torch.Tensor):
    tri = vertices[faces]
    cr = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    area2 = cr.norm(dim=1)  # 2 * area
    normals = cr / (area2.unsqueeze(1) + 1e-20)
    return area2, normals


def triangleInversionBarrierLoss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    ref_normals: torch.Tensor,
    ref_area2: torch.Tensor,
    flip_margin: float = 0.0,
    area_frac: float = 0.1,
) -> torch.Tensor:
    """Penalize normal flips and area collapse relative to a rest reference.

    Args:
        ref_normals: (F, 3) detached rest unit normals.
        ref_area2: (F,) detached rest 2*area.
        flip_margin: barrier activates when n . n_ref < flip_margin (default 0
            = penalize only actual flips). Raise toward 1 to keep normals stiff.
        area_frac: barrier activates when area drops below area_frac * rest area.

    Returns:
        scalar loss (mean over faces).
    """
    area2, normals = triangleAreaNormals(vertices, faces)
    align = (normals * ref_normals).sum(dim=1)
    flip_viol = torch.clamp(flip_margin - align, min=0.0)
    area_viol = torch.clamp(area_frac * ref_area2 - area2, min=0.0) / (
        ref_area2 + 1e-20
    )
    return (flip_viol ** 2).mean() + (area_viol ** 2).mean()
