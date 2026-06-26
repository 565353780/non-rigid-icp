"""Trajectory-based self-intersection guard for projection fitting.

Core invariant (first principles): the watertight source is intersection-free, so
if every vertex is moved along the STRAIGHT segment from its rest position on the
watertight mesh (`ref`) to a proposed position (`v'`) and no such segment pierces
any non-incident face of the current mesh, then no vertex has tunnelled across a
sheet -- the dominant double-layer collapse failure mode is eliminated. This is a
necessary condition that is cheap and embarrassingly parallel (AABB broad phase +
segment-triangle narrow phase); the fitter keeps the authoritative
`findSelfIntersections` scan as the final hard gate.

The broad phase reuses the same uniform grid hash as the collision / self-
intersection scan (`Method/spatial_hash.py`): segments and triangles are placed
in ONE combined AABB list, within-cell pairs are streamed in bounded chunks, and
only the (segment, triangle) cross pairs are kept. Candidate generation runs once
per resolve; the bisection then only re-runs the (cheap) narrow phase on that
fixed candidate set, because shrinking the step only shrinks each segment's AABB
(its candidate triangles stay a subset).
"""

import torch
from typing import Tuple, Union

from non_rigid_icp.Method.geometry import segmentTriangleIntersect
from non_rigid_icp.Method.spatial_hash import (
    triangleAABBs,
    estimateCellSize,
    buildCellIncidences,
    cellPairChunk,
    aabbOverlap,
)


def _segment_aabbs(
    seg_start: torch.Tensor, seg_end: torch.Tensor, inflate: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    lo = torch.minimum(seg_start, seg_end) - inflate
    hi = torch.maximum(seg_start, seg_end) + inflate
    return lo, hi


def segmentMeshCandidates(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    owner_vid: torch.Tensor,
    inflate: float = 0.0,
    cell_size: Union[float, None] = None,
    pair_chunk: int = 40_000_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broad-phase (segment, triangle) candidate pairs, owner-1-ring excluded.

    Args:
        seg_start/seg_end: (S, 3) segment endpoints (rest -> proposed).
        vertices/faces: the current mesh the segments are tested against.
        owner_vid: (S,) vertex id each segment belongs to; faces incident to it
            (the 1-ring, which legitimately share the moving endpoint) are
            dropped from the candidates.
        inflate: AABB inflation (small, for conservativeness).

    Returns:
        cand_seg: (P,) local segment ids; cand_tri: (P,) global face ids.
    """
    device = faces.device
    s = seg_start.shape[0]
    f = faces.shape[0]
    if s == 0 or f == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z

    seg_lo, seg_hi = _segment_aabbs(seg_start, seg_end, inflate)
    tri_lo, tri_hi = triangleAABBs(vertices, faces, inflate)

    # Size the grid from the TRIANGLE boxes only. The combined list places the
    # (tiny, capped) segments first, and estimateCellSize samples the first 4M
    # boxes -- sizing on segments would pick a cell far finer than the triangles
    # and explode the incidence count (28.9M faces x hundreds of cells -> OOM).
    if cell_size is None:
        cell_size = estimateCellSize(tri_lo, tri_hi, quantile=0.9, factor=1.0)
        if inflate > 0:
            cell_size = max(cell_size, 2.0 * inflate)

    lo = torch.cat([seg_lo, tri_lo], dim=0)
    hi = torch.cat([seg_hi, tri_hi], dim=0)

    tri_sorted, pair_off, total_pairs = buildCellIncidences(lo, hi, cell_size)
    seg_parts = []
    tri_parts = []
    for start in range(0, total_pairs, pair_chunk):
        count = min(pair_chunk, total_pairs - start)
        a, b = cellPairChunk(tri_sorted, pair_off, start, count)
        keep = a != b
        a, b = a[keep], b[keep]
        if a.numel() == 0:
            continue
        ov = aabbOverlap(lo, hi, a, b)
        a, b = a[ov], b[ov]
        if a.numel() == 0:
            continue
        # keep only cross pairs: exactly one endpoint is a segment (< s)
        a_seg = a < s
        b_seg = b < s
        cross = a_seg ^ b_seg
        a, b, a_seg = a[cross], b[cross], a_seg[cross]
        if a.numel() == 0:
            continue
        seg_id = torch.where(a_seg, a, b)
        tri_id = torch.where(a_seg, b, a) - s
        # drop faces incident to the segment's owner vertex (the moving 1-ring)
        owner = owner_vid[seg_id]
        incident = (faces[tri_id] == owner.unsqueeze(1)).any(dim=1)
        seg_id, tri_id = seg_id[~incident], tri_id[~incident]
        if seg_id.numel() == 0:
            continue
        seg_parts.append(seg_id)
        tri_parts.append(tri_id)

    if not seg_parts:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    return torch.cat(seg_parts, dim=0), torch.cat(tri_parts, dim=0)


def _narrow_seg_hits(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    cand_seg: torch.Tensor,
    cand_tri: torch.Tensor,
    n_seg: int,
    narrow_chunk: int = 8_000_000,
) -> torch.Tensor:
    """Per-segment bool: does any candidate triangle pierce this segment?"""
    out = torch.zeros(n_seg, dtype=torch.bool, device=faces.device)
    p = cand_seg.shape[0]
    for start in range(0, p, narrow_chunk):
        cs = cand_seg[start : start + narrow_chunk]
        ct = cand_tri[start : start + narrow_chunk]
        tri = faces[ct]
        hit = segmentTriangleIntersect(
            seg_start[cs],
            seg_end[cs],
            vertices[tri[:, 0]],
            vertices[tri[:, 1]],
            vertices[tri[:, 2]],
        )
        if bool(hit.any()):
            out[cs[hit]] = True
    return out


def segmentsCrossMesh(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    owner_vid: torch.Tensor,
    inflate: float = 0.0,
) -> torch.Tensor:
    """(S,) bool: does segment [start, end] pierce any non-incident face?"""
    cand_seg, cand_tri = segmentMeshCandidates(
        seg_start, seg_end, vertices, faces, owner_vid, inflate=inflate
    )
    return _narrow_seg_hits(
        seg_start, seg_end, vertices, faces, cand_seg, cand_tri, seg_start.shape[0]
    )


def largestSafeStep(
    ref: torch.Tensor,
    proposed: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    owner_vid: torch.Tensor,
    inflate: float = 0.0,
    n_bisect: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Largest safe position along [ref, proposed] (per segment).

    Returns:
        safe_pos: (S, 3) = ref + alpha * (proposed - ref), alpha the largest
            tested fraction whose segment pierces no non-incident face.
        clamped: (S,) bool, True where alpha < 1 (the step was pulled back).
    """
    s = ref.shape[0]
    if s == 0:
        return proposed, torch.zeros(0, dtype=torch.bool, device=faces.device)

    cand_seg, cand_tri = segmentMeshCandidates(
        ref, proposed, vertices, faces, owner_vid, inflate=inflate
    )
    full_hit = _narrow_seg_hits(
        ref, proposed, vertices, faces, cand_seg, cand_tri, s
    )
    if not bool(full_hit.any()):
        return proposed, torch.zeros(s, dtype=torch.bool, device=faces.device)

    # only the crossing segments need bisection; restrict candidates to them
    need = full_hit
    m = need[cand_seg]
    cs, ct = cand_seg[m], cand_tri[m]

    direction = proposed - ref
    lo = torch.zeros(s, device=ref.device)  # alpha=0 (degenerate point) is safe
    hi = torch.ones(s, device=ref.device)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        endpt = ref + mid.unsqueeze(1) * direction
        hit = _narrow_seg_hits(ref, endpt, vertices, faces, cs, ct, s)
        lo = torch.where(need & (~hit), mid, lo)
        hi = torch.where(need & hit, mid, hi)

    alpha = torch.where(need, lo, torch.ones_like(lo))
    safe_pos = ref + alpha.unsqueeze(1) * direction
    return safe_pos, need
