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

from non_rigid_icp.Method.triton_kernels import segmentTrianglePairHits
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
    face_ids: Union[torch.Tensor, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broad-phase (segment, triangle) candidate pairs, owner-1-ring excluded.

    Args:
        seg_start/seg_end: (S, 3) segment endpoints (rest -> proposed).
        vertices/faces: the current mesh the segments are tested against.
        owner_vid: (S,) vertex id each segment belongs to; faces incident to it
            (the 1-ring, which legitimately share the moving endpoint) are
            dropped from the candidates.
        inflate: AABB inflation (small, for conservativeness).
        cell_size: optional fixed grid cell size; caching it across calls (it is
            tessellation-scale and stable within a topology) avoids the per-call
            quantile estimate over the full triangle AABB set.
        face_ids: optional (M,) subset of GLOBAL face ids to test against. The
            returned `cand_tri` is always remapped back to GLOBAL ids, so the
            narrow phase keeps using the full `faces`/`vertices`. The default
            (None) tests against every face, which never misses a static
            opposing sheet -- restrict only when the partner is provably inside
            the subset (e.g. an already-dilated active region).

    Returns:
        cand_seg: (P,) local segment ids; cand_tri: (P,) global face ids.
    """
    device = faces.device
    s = seg_start.shape[0]
    local_faces = faces if face_ids is None else faces[face_ids]
    f = local_faces.shape[0]
    if s == 0 or f == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z

    seg_lo, seg_hi = _segment_aabbs(seg_start, seg_end, inflate)
    tri_lo, tri_hi = triangleAABBs(vertices, local_faces, inflate)

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
        tri_id = torch.where(a_seg, b, a) - s  # LOCAL index into local_faces
        # drop faces incident to the segment's owner vertex (the moving 1-ring)
        owner = owner_vid[seg_id]
        incident = (local_faces[tri_id] == owner.unsqueeze(1)).any(dim=1)
        seg_id, tri_id = seg_id[~incident], tri_id[~incident]
        if seg_id.numel() == 0:
            continue
        seg_parts.append(seg_id)
        tri_parts.append(tri_id)

    if not seg_parts:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    cand_seg = torch.cat(seg_parts, dim=0)
    cand_tri = torch.cat(tri_parts, dim=0)
    if face_ids is not None:
        cand_tri = face_ids[cand_tri]  # remap LOCAL -> GLOBAL
    return cand_seg, cand_tri


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
    if p == 0:
        return out
    # gather the 3 triangle-vertex arrays ONCE (the Triton pair kernel indexes
    # them by global face id, fusing the per-pair gather + Moller-Trumbore math).
    va = vertices[faces[:, 0]]
    vb = vertices[faces[:, 1]]
    vc = vertices[faces[:, 2]]
    for start in range(0, p, narrow_chunk):
        cs = cand_seg[start : start + narrow_chunk]
        ct = cand_tri[start : start + narrow_chunk]
        hit = segmentTrianglePairHits(
            cs, ct, seg_start, seg_end, va, vb, vc
        )
        if bool(hit.any()):
            out[cs[hit]] = True
    return out


def _narrow_seg_pairs(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    cand_seg: torch.Tensor,
    cand_tri: torch.Tensor,
    narrow_chunk: int = 8_000_000,
) -> torch.Tensor:
    """Per-candidate-pair bool: does this (segment, triangle) pair pierce?

    Unlike `_narrow_seg_hits` (which collapses to a per-segment OR for the cheap
    boolean / bisection path), this keeps the result aligned with the candidate
    pair arrays so the caller can recover the exact (segment, face) hits."""
    p = cand_seg.shape[0]
    if p == 0:
        return torch.zeros(0, dtype=torch.bool, device=faces.device)
    out = torch.zeros(p, dtype=torch.bool, device=faces.device)
    va = vertices[faces[:, 0]]
    vb = vertices[faces[:, 1]]
    vc = vertices[faces[:, 2]]
    for start in range(0, p, narrow_chunk):
        sl = slice(start, start + narrow_chunk)
        cs = cand_seg[sl]
        ct = cand_tri[sl]
        out[sl] = segmentTrianglePairHits(
            cs, ct, seg_start, seg_end, va, vb, vc
        )
    return out


def segmentsCrossMesh(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    owner_vid: torch.Tensor,
    inflate: float = 0.0,
    cell_size: Union[float, None] = None,
    face_ids: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """(S,) bool: does segment [start, end] pierce any non-incident face?"""
    cand_seg, cand_tri = segmentMeshCandidates(
        seg_start,
        seg_end,
        vertices,
        faces,
        owner_vid,
        inflate=inflate,
        cell_size=cell_size,
        face_ids=face_ids,
    )
    return _narrow_seg_hits(
        seg_start, seg_end, vertices, faces, cand_seg, cand_tri, seg_start.shape[0]
    )


def segmentMeshIntersections(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    owner_vid: torch.Tensor,
    inflate: float = 0.0,
    cell_size: Union[float, None] = None,
    face_ids: Union[torch.Tensor, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detailed segment-mesh crossings.

    This is the authoritative form of the user's self-intersection criterion: a
    fitted vertex's straight trajectory from its watertight-rest position
    (`seg_start`) to its current fitted position (`seg_end`) must not pierce any
    non-incident face of the current mesh.

    Returns:
        hit_seg: (P,) local segment ids that pierced a face.
        hit_face: (P,) GLOBAL face ids each `hit_seg` pierced (paired arrays).
        crossed: (S,) bool, True where the segment pierced at least one face.
    """
    s = seg_start.shape[0]
    z = torch.zeros(0, dtype=torch.long, device=faces.device)
    crossed = torch.zeros(s, dtype=torch.bool, device=faces.device)
    cand_seg, cand_tri = segmentMeshCandidates(
        seg_start,
        seg_end,
        vertices,
        faces,
        owner_vid,
        inflate=inflate,
        cell_size=cell_size,
        face_ids=face_ids,
    )
    if cand_seg.numel() == 0:
        return z, z, crossed
    pair_hit = _narrow_seg_pairs(
        seg_start, seg_end, vertices, faces, cand_seg, cand_tri
    )
    hit_seg = cand_seg[pair_hit]
    hit_face = cand_tri[pair_hit]
    crossed[hit_seg] = True
    return hit_seg, hit_face, crossed


def largestSafeStep(
    ref: torch.Tensor,
    proposed: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    owner_vid: torch.Tensor,
    inflate: float = 0.0,
    n_bisect: int = 6,
    cell_size: Union[float, None] = None,
    face_ids: Union[torch.Tensor, None] = None,
    return_hits: bool = False,
):
    """Largest safe position along [ref, proposed] (per segment).

    Returns:
        safe_pos: (S, 3) = ref + alpha * (proposed - ref), alpha the largest
            tested fraction whose segment pierces no non-incident face.
        clamped: (S,) bool, True where alpha < 1 (the step was pulled back).
        (if return_hits) hit_seg, hit_face: the (segment, GLOBAL face) pairs that
            the FULL ref->proposed trajectory pierced -- the offending region the
            caller can feed back into a barrier.
    """
    s = ref.shape[0]
    z = torch.zeros(0, dtype=torch.long, device=faces.device)
    if s == 0:
        empty_bool = torch.zeros(0, dtype=torch.bool, device=faces.device)
        if return_hits:
            return proposed, empty_bool, z, z
        return proposed, empty_bool

    cand_seg, cand_tri = segmentMeshCandidates(
        ref,
        proposed,
        vertices,
        faces,
        owner_vid,
        inflate=inflate,
        cell_size=cell_size,
        face_ids=face_ids,
    )
    pair_hit = _narrow_seg_pairs(ref, proposed, vertices, faces, cand_seg, cand_tri)
    full_hit = torch.zeros(s, dtype=torch.bool, device=faces.device)
    if cand_seg.numel() > 0:
        full_hit[cand_seg[pair_hit]] = True

    if not bool(full_hit.any()):
        empty_bool = torch.zeros(s, dtype=torch.bool, device=ref.device)
        if return_hits:
            return proposed, empty_bool, z, z
        return proposed, empty_bool

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
    if return_hits:
        return safe_pos, need, cand_seg[pair_hit], cand_tri[pair_hit]
    return safe_pos, need
