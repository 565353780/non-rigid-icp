"""Self-collision detection for the non-rigid fit.

Two-phase, scalable to tens of millions of faces:

  broad phase  - centroid k-NN (reusing the GPU spatial index) yields a small
                 set of geometrically-near, topologically-non-adjacent face
                 pairs (a fold brings opposing sheets within k nearest).
  narrow phase - exact triangle-triangle intersection / distance on that small
                 candidate set (vectorized, see Method/geometry.py).

The fitter uses this both as a hard guard ("did this step create a NEW
self-intersection?") and, via Loss/collision.py, as a soft barrier that keeps
non-adjacent sheets separated before they cross.
"""

import torch
import numpy as np
from typing import Tuple, Union

from non_rigid_icp.Method.geometry import (
    triangleTriangleIntersects,
    triangleTriangleDistance2,
)
from non_rigid_icp.Method.topology import facePairsShareVertex


def triangleCentroidsAndRadii(
    vertices: torch.Tensor, faces: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-face centroid (F,3) and bounding radius (F,) = centroid->farthest vertex."""
    tri = vertices[faces]  # (F, 3, 3)
    centroids = tri.mean(dim=1)
    radii = ((tri - centroids.unsqueeze(1)) ** 2).sum(dim=-1).sqrt().max(dim=1).values
    return centroids, radii


def gatherTrianglePairs(
    vertices: torch.Tensor, faces: torch.Tensor, pairs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather vertex coordinates for face pairs: (tri1, tri2) each (P, 3, 3)."""
    tri1 = vertices[faces[pairs[:, 0]]]
    tri2 = vertices[faces[pairs[:, 1]]]
    return tri1, tri2


def pairKeys(pairs: torch.Tensor, num_faces: int) -> torch.Tensor:
    """Stable order-independent integer key per face pair (min*F + max)."""
    lo = torch.minimum(pairs[:, 0], pairs[:, 1]).to(torch.int64)
    hi = torch.maximum(pairs[:, 0], pairs[:, 1]).to(torch.int64)
    return lo * int(num_faces) + hi


def buildCollisionCandidates(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    k: int = 8,
    margin: float = 0.0,
    active_face_mask: Union[torch.Tensor, None] = None,
    device: str = "cuda",
    chunk: int = 1000000,
) -> torch.Tensor:
    """Broad-phase candidate face pairs that may self-collide.

    For each (active) face we take its k nearest faces by centroid, keep pairs
    whose bounding spheres overlap within `margin`, drop topologically-adjacent
    pairs (sharing a vertex), and de-duplicate.

    Args:
        vertices: (V, 3) float tensor.
        faces: (F, 3) long tensor.
        k: neighbours per face in the broad phase.
        margin: extra separation added to the sphere-overlap test.
        active_face_mask: (F,) bool; if given, only these faces issue queries
            (neighbours are still searched over ALL faces). Bounds the work to
            regions that actually moved.
        chunk: query-face chunk size to bound peak memory.

    Returns:
        pairs: (P, 2) long, P typically tiny relative to F.
    """
    from non_rigid_icp.Method.nn import NNIndex

    f = faces.shape[0]
    centroids, radii = triangleCentroidsAndRadii(vertices, faces)
    centroids_np = centroids.detach().cpu().numpy().astype(np.float32)
    index = NNIndex(centroids_np, device=device)

    if active_face_mask is None:
        query_ids = torch.arange(f, device=faces.device)
    else:
        query_ids = torch.nonzero(active_face_mask, as_tuple=False).reshape(-1)
    if query_ids.numel() == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=faces.device)

    k_eff = min(k + 1, f)  # +1 because the nearest centroid is the face itself
    collected = []
    for start in range(0, query_ids.numel(), chunk):
        qid = query_ids[start : start + chunk]
        q_np = centroids_np[qid.detach().cpu().numpy()]
        nbr_idx, _ = index.queryKNN(q_np, k_eff)  # (n, k_eff)
        nbr = torch.from_numpy(nbr_idx).to(faces.device)  # (n, k_eff)
        src = qid.unsqueeze(1).expand_as(nbr)  # (n, k_eff)

        a = src.reshape(-1)
        b = nbr.reshape(-1)
        valid = a != b
        a, b = a[valid], b[valid]
        if a.numel() == 0:
            continue

        # bounding-sphere overlap test
        cd = (centroids[a] - centroids[b]).pow(2).sum(dim=-1).sqrt()
        overlap = cd < (radii[a] + radii[b] + margin)
        a, b = a[overlap], b[overlap]
        if a.numel() == 0:
            continue

        pr = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], dim=1)
        # drop topologically adjacent pairs (share a vertex)
        shared = facePairsShareVertex(pr, faces)
        pr = pr[~shared]
        if pr.numel() > 0:
            collected.append(pr)

    if not collected:
        return torch.zeros(0, 2, dtype=torch.long, device=faces.device)

    pairs = torch.cat(collected, dim=0)
    pairs = torch.unique(pairs, dim=0)
    return pairs


def detectIntersectingPairs(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pairs: torch.Tensor,
    chunk: int = 2000000,
) -> torch.Tensor:
    """Boolean mask over `pairs`: which candidate face pairs actually intersect."""
    p = pairs.shape[0]
    if p == 0:
        return torch.zeros(0, dtype=torch.bool, device=faces.device)
    out = torch.zeros(p, dtype=torch.bool, device=faces.device)
    for start in range(0, p, chunk):
        sub = pairs[start : start + chunk]
        tri1, tri2 = gatherTrianglePairs(vertices, faces, sub)
        out[start : start + chunk] = triangleTriangleIntersects(tri1, tri2)
    return out


def detectNewSelfIntersections(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pairs: torch.Tensor,
    baseline_keys: Union[torch.Tensor, None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find intersecting candidate pairs that are NOT in the baseline ignore set.

    Args:
        baseline_keys: (B,) int64 stable keys of pairs allowed to intersect
            (already-touching / degenerate at init). May be None/empty.

    Returns:
        new_pairs: (Q, 2) intersecting pairs absent from baseline.
        intersecting_keys: (M,) keys of ALL currently intersecting pairs.
    """
    f = faces.shape[0]
    if pairs.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=faces.device)
        return torch.zeros(0, 2, dtype=torch.long, device=faces.device), empty

    hit = detectIntersectingPairs(vertices, faces, pairs)
    inter_pairs = pairs[hit]
    inter_keys = pairKeys(inter_pairs, f)

    if baseline_keys is None or baseline_keys.numel() == 0:
        return inter_pairs, inter_keys

    is_baseline = torch.isin(inter_keys, baseline_keys.to(inter_keys.device))
    new_pairs = inter_pairs[~is_baseline]
    return new_pairs, inter_keys
