import torch
import numpy as np
from typing import Tuple, Union

if torch.cuda.is_available():
    try:
        from non_rigid_icp.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
    except Exception as _e:
        print("[WARN][Metric.chamfer] CUDA chamfer op unavailable, using fallback:", _e)
        chamfer_3DDist = None
else:
    chamfer_3DDist = None


@torch.no_grad()
def toL1ChamferDistance(dist1: torch.Tensor, dist2: torch.Tensor) -> float:
    dist1 = torch.sqrt(dist1.detach().clone())
    dist2 = torch.sqrt(dist2.detach().clone())

    mean_dist1 = torch.mean(dist1)
    mean_dist2 = torch.mean(dist2)

    l1_chamfer = mean_dist1 + mean_dist2

    return l1_chamfer.item()

@torch.no_grad()
def toL2ChamferDistance(dist1: torch.Tensor, dist2: torch.Tensor) -> float:
    dist1 = dist1.detach().clone()
    dist2 = dist2.detach().clone()

    mean_dist1 = torch.mean(dist1)
    mean_dist2 = torch.mean(dist2)

    l2_chamfer = mean_dist1 + mean_dist2

    return l2_chamfer.item()


def _toPointTensor(points: Union[np.ndarray, torch.Tensor], device: str) -> torch.Tensor:
    if isinstance(points, torch.Tensor):
        t = points.detach().to(device=device, dtype=torch.float32)
    else:
        t = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t.contiguous()


@torch.no_grad()
def _chunkedNearestSquaredDist(
    query: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int = 40000,
) -> torch.Tensor:
    """Squared distance from each query point to its nearest reference point.

    Pure-torch brute force in chunks; works on CPU or GPU without the custom
    CUDA chamfer op. query: (Nq,3), reference: (Nr,3). Returns (Nq,).
    """
    nq = query.shape[0]
    out = torch.empty(nq, device=query.device, dtype=torch.float32)
    ref_sq = (reference * reference).sum(dim=1)  # (Nr,)
    for start in range(0, nq, chunk_size):
        end = min(start + chunk_size, nq)
        q = query[start:end]  # (c, 3)
        q_sq = (q * q).sum(dim=1, keepdim=True)  # (c, 1)
        # (c, Nr) squared dist = |q|^2 - 2 q.r + |r|^2
        d = q_sq + ref_sq.unsqueeze(0) - 2.0 * (q @ reference.t())
        out[start:end] = torch.clamp(d.min(dim=1).values, min=0.0)
    return out


@torch.no_grad()
def nearestSquaredDistances(
    source_points: Union[np.ndarray, torch.Tensor],
    target_points: Union[np.ndarray, torch.Tensor],
    device: Union[str, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional nearest-neighbor squared distances.

    Returns (dist_s2t, dist_t2s), each 1-D squared-distance tensors:
        dist_s2t[i] = min_j |source_i - target_j|^2
        dist_t2s[j] = min_i |target_j - source_i|^2

    Uses a spatial NN index (Open3D, GPU when available) so this scales to
    millions of points. Falls back to the custom chamfer op, then to a chunked
    brute-force implementation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # preferred: spatial index (O(N log M)); handles millions of points
    try:
        from non_rigid_icp.Method.nn import NNIndex

        s_np = (
            source_points.detach().cpu().numpy()
            if isinstance(source_points, torch.Tensor)
            else np.asarray(source_points)
        )
        t_np = (
            target_points.detach().cpu().numpy()
            if isinstance(target_points, torch.Tensor)
            else np.asarray(target_points)
        )
        t_index = NNIndex(t_np, device=device)
        _, d_s2t = t_index.query(s_np, k=1)
        s_index = NNIndex(s_np, device=device)
        _, d_t2s = s_index.query(t_np, k=1)
        return (
            torch.from_numpy(d_s2t).to(device),
            torch.from_numpy(d_t2s).to(device),
        )
    except Exception as e:
        print("[WARN][nearestSquaredDistances] NN index failed, fallback:", e)

    s = _toPointTensor(source_points, device)
    t = _toPointTensor(target_points, device)

    if device != "cpu" and chamfer_3DDist is not None:
        try:
            dist1, dist2 = chamfer_3DDist()(s, t)[:2]
            return dist1.squeeze(0), dist2.squeeze(0)
        except Exception as e:
            print("[WARN][nearestSquaredDistances] chamfer op failed, fallback:", e)

    s2 = s.squeeze(0)
    t2 = t.squeeze(0)
    dist_s2t = _chunkedNearestSquaredDist(s2, t2)
    dist_t2s = _chunkedNearestSquaredDist(t2, s2)
    return dist_s2t, dist_t2s


@torch.no_grad()
def computeChamferMetrics(
    source_points: Union[np.ndarray, torch.Tensor],
    target_points: Union[np.ndarray, torch.Tensor],
    device: Union[str, None] = None,
) -> dict:
    """Compute bidirectional Chamfer metrics.

    Returns a dict with squared (L2) and rooted (L1) errors:
        fit_error_l2 / fit_error_l1: source -> target (how well source sits on target)
        cov_error_l2 / cov_error_l1: target -> source (how well source covers target)
        chamfer_l1 / chamfer_l2: symmetric sums
    """
    dist_s2t, dist_t2s = nearestSquaredDistances(source_points, target_points, device)

    fit_l2 = torch.mean(dist_s2t).item()
    cov_l2 = torch.mean(dist_t2s).item()
    fit_l1 = torch.mean(torch.sqrt(torch.clamp(dist_s2t, min=0.0))).item()
    cov_l1 = torch.mean(torch.sqrt(torch.clamp(dist_t2s, min=0.0))).item()

    return {
        "fit_error_l2": fit_l2,
        "cov_error_l2": cov_l2,
        "chamfer_l2": fit_l2 + cov_l2,
        "fit_error_l1": fit_l1,
        "cov_error_l1": cov_l1,
        "chamfer_l1": fit_l1 + cov_l1,
    }


@torch.no_grad()
def computeF1AtThreshold(
    source_points: Union[np.ndarray, torch.Tensor],
    target_points: Union[np.ndarray, torch.Tensor],
    tau: float,
    device: Union[str, None] = None,
) -> dict:
    """F1-Score at distance threshold tau.

    precision = fraction of source points within tau of target surface
    recall    = fraction of target points within tau of source surface
    f1        = 2 * P * R / (P + R)

    tau must be expressed in the SAME coordinate frame as the input points.
    """
    dist_s2t, dist_t2s = nearestSquaredDistances(source_points, target_points, device)

    tau2 = float(tau) ** 2
    precision = (dist_s2t < tau2).float().mean().item()
    recall = (dist_t2s < tau2).float().mean().item()

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "tau": float(tau),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
