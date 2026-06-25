import torch
from typing import Tuple, Union

try:
    from non_rigid_icp.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist

    _CHAMFER = chamfer_3DDist()
except Exception as _e:
    print("[WARN][Loss.surface] CUDA chamfer op unavailable:", _e)
    _CHAMFER = None


def _nn(query: torch.Tensor, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest neighbor of query in reference.

    query/reference: (N,3)/(M,3). Returns (sq_dist (N,), idx (N,)).
    Uses the custom CUDA chamfer op; falls back to chunked brute force.
    """
    if _CHAMFER is not None and query.is_cuda:
        d1, _, i1, _ = _CHAMFER(query.unsqueeze(0).contiguous(), reference.unsqueeze(0).contiguous())
        return d1.squeeze(0), i1.squeeze(0).long()

    chunk = 20000
    n = query.shape[0]
    out_d = torch.empty(n, device=query.device)
    out_i = torch.empty(n, dtype=torch.long, device=query.device)
    ref_sq = (reference * reference).sum(1)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        q = query[s:e]
        d = (q * q).sum(1, keepdim=True) + ref_sq.unsqueeze(0) - 2.0 * (q @ reference.t())
        md, mi = torch.clamp(d, min=0.0).min(1)
        out_d[s:e] = md
        out_i[s:e] = mi
    return out_d, out_i


def maskedDistLoss(
    source_points: torch.Tensor,
    matched_points: torch.Tensor,
    max_dist: Union[float, None] = None,
) -> torch.Tensor:
    """Mean squared distance between source points and their matched targets.

    Points farther than max_dist (if given) are masked out (outlier rejection).
    """
    d2 = ((source_points - matched_points) ** 2).sum(dim=1)
    if max_dist is not None and max_dist != float("inf"):
        mask = d2 < max_dist ** 2
        if mask.any():
            d2 = d2[mask]
    return d2.mean()


def symmetricChamferLoss(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    max_dist: Union[float, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric chamfer: source->target fit and target->source coverage.

    Both source_points and target_points are (N,3) / (M,3). The source->target
    term is differentiable w.r.t. source_points (via matched target points,
    treated as constants); the target->source term is differentiable w.r.t. the
    source points matched to each target point.

    Returns (fit_loss, coverage_loss), both mean-squared distances with optional
    outlier masking at max_dist.
    """
    d_s2t, idx_s2t = _nn(source_points, target_points)
    matched_t = target_points[idx_s2t]
    fit_loss = maskedDistLoss(source_points, matched_t, max_dist)

    d_t2s, idx_t2s = _nn(target_points, source_points)
    matched_s = source_points[idx_t2s]
    cov_loss = maskedDistLoss(matched_s, target_points, max_dist)

    return fit_loss, cov_loss


def pointToPlaneLoss(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    max_dist: Union[float, None] = None,
) -> torch.Tensor:
    """Point-to-plane loss: squared projection of (source - matched_target) onto
    the matched target normal. More accurate for final surface snapping near
    edges and thin structures.
    """
    d2, idx = _nn(source_points, target_points)
    matched_t = target_points[idx]
    matched_n = target_normals[idx]
    diff = source_points - matched_t
    plane_dist = (diff * matched_n).sum(dim=1)
    pd2 = plane_dist ** 2
    if max_dist is not None and max_dist != float("inf"):
        mask = d2 < max_dist ** 2
        if mask.any():
            pd2 = pd2[mask]
    return pd2.mean()


def edgeLaplacianLoss(
    displacement: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Smoothness on the per-vertex displacement field.

    Penalizes the squared difference of displacement across each edge, which
    keeps neighboring vertices moving coherently (as-rigid-as-possible-ish on
    the displacement field). edges: (E,2) long.
    """
    diff = displacement[edges[:, 0]] - displacement[edges[:, 1]]
    return (diff ** 2).sum(dim=1).mean()
