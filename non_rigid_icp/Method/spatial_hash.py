"""Uniform spatial-hash broad phase for triangle pairs.

Unlike a centroid k-NN broad phase (which can miss the opposing sheet of a thin
/ double-layer structure when same-sheet neighbours fill the k slots), a grid
hash over INFLATED triangle AABBs is *complete*: if two (inflated) AABBs
overlap they necessarily share a grid cell, so the pair is always generated.

The number of within-cell candidate pairs can reach billions in densely folded
/ already-self-intersecting regions (thousands of triangles per cell), so the
pairs are produced as a *stream* of bounded-size chunks rather than a single
allocation, and a cheap real-AABB overlap test prunes most of them before the
exact triangle-triangle test runs. Everything is vectorized (no python
per-element loops over faces).
"""

import torch
from typing import Tuple, Union, Iterator


def _exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    if x.numel() > 1:
        out[1:] = torch.cumsum(x, dim=0)[:-1]
    return out


def triangleAABBs(
    vertices: torch.Tensor, faces: torch.Tensor, inflate: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-triangle axis-aligned bounding boxes, optionally inflated by `inflate`.

    Returns (lo, hi), each (F, 3).
    """
    tri = vertices[faces]  # (F, 3, 3)
    lo = tri.amin(dim=1) - inflate
    hi = tri.amax(dim=1) + inflate
    return lo, hi


def estimateCellSize(
    lo: torch.Tensor, hi: torch.Tensor, quantile: float = 0.9, factor: float = 1.0
) -> float:
    """Pick a grid cell size from the triangle AABB extents.

    A cell ~ the (quantile) triangle extent keeps the cells-per-triangle O(1).
    Triangles-per-cell is then governed by the local surface layering, which is
    handled by the streaming pair generation.
    """
    extent = (hi - lo).amax(dim=1)
    extent = extent[extent > 0]
    if extent.numel() == 0:
        return 1.0
    sample = extent if extent.numel() < 4_000_000 else extent[:4_000_000]
    q = torch.quantile(sample, quantile).item()
    return max(q * factor, 1e-9)


def buildCellIncidences(
    lo: torch.Tensor, hi: torch.Tensor, cell_size: float
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Rasterize triangle AABBs into a uniform grid and prepare pair streaming.

    Returns:
        tri_sorted: (N,) triangle id per (triangle, cell) incidence, ordered so
            incidences sharing a cell are contiguous.
        pair_off: (N,) exclusive cumulative count of within-cell partner pairs
            per incidence (used to map a global pair index back to a position).
        total_pairs: total number of unordered within-cell candidate pairs.
    """
    device = lo.device
    f = lo.shape[0]
    if f == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z, 0

    cmin = torch.floor(lo / cell_size).to(torch.int64)
    cmax = torch.floor(hi / cell_size).to(torch.int64)
    span = (cmax - cmin + 1).clamp(min=1)
    n_cells = span.prod(dim=1)

    total = int(n_cells.sum().item())
    tri_rep = torch.repeat_interleave(torch.arange(f, device=device), n_cells)
    offsets = _exclusive_cumsum(n_cells)
    li = torch.arange(total, device=device) - offsets[tri_rep]

    sx = span[:, 0][tri_rep]
    sy = span[:, 1][tri_rep]
    dx = li % sx
    dy = (li // sx) % sy
    dz = li // (sx * sy)
    cx = cmin[:, 0][tri_rep] + dx
    cy = cmin[:, 1][tri_rep] + dy
    cz = cmin[:, 2][tri_rep] + dz

    cx = cx - cx.min()
    cy = cy - cy.min()
    cz = cz - cz.min()
    dimy = int(cy.max().item()) + 1
    dimz = int(cz.max().item()) + 1
    key = (cx * dimy + cy) * dimz + cz
    # free the large per-incidence intermediates before the argsort workspace
    del cmin, cmax, span, offsets, li, sx, sy, dx, dy, dz, cx, cy, cz

    order = torch.argsort(key)
    key_s = key[order]
    tri_sorted = tri_rep[order]
    del key, tri_rep, order

    is_new = torch.ones(total, dtype=torch.bool, device=device)
    is_new[1:] = key_s[1:] != key_s[:-1]
    seg_id = torch.cumsum(is_new.long(), dim=0) - 1
    seg_start_idx = torch.nonzero(is_new, as_tuple=False).reshape(-1)
    seg_end_idx = torch.cat(
        [seg_start_idx[1:], torch.tensor([total], device=device)]
    )
    seg_len = seg_end_idx - seg_start_idx  # (C,)

    lidx = torch.arange(total, device=device) - seg_start_idx[seg_id]
    n_partners = seg_len[seg_id] - lidx - 1  # partners with higher position
    pair_off = _exclusive_cumsum(n_partners)
    total_pairs = int(n_partners.sum().item())
    return tri_sorted, pair_off, total_pairs


def cellPairChunk(
    tri_sorted: torch.Tensor, pair_off: torch.Tensor, start: int, count: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode global pair indices [start, start+count) into triangle id pairs.

    Within-cell partners always have a higher position in `tri_sorted`, and a
    cell's positions are contiguous, so `src+1+local` is guaranteed to stay
    inside the same cell segment (no segment bookkeeping needed here).
    """
    device = tri_sorted.device
    kk = torch.arange(start, start + count, device=device)
    src_pos = torch.searchsorted(pair_off, kk, right=True) - 1
    local = kk - pair_off[src_pos]
    partner_pos = src_pos + 1 + local
    a = tri_sorted[src_pos]
    b = tri_sorted[partner_pos]
    return a, b


def aabbOverlap(
    lo: torch.Tensor, hi: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Cheap exact AABB-overlap mask for triangle id pairs (a, b)."""
    return (lo[a] <= hi[b]).all(dim=1) & (lo[b] <= hi[a]).all(dim=1)


def streamOverlapPairs(
    lo: torch.Tensor,
    hi: torch.Tensor,
    cell_size: float,
    pair_chunk: int = 40_000_000,
) -> Iterator[torch.Tensor]:
    """Yield (P_i, 2) chunks of AABB-overlapping triangle pairs (a < b).

    Memory is bounded by `pair_chunk` regardless of how many billions of
    within-cell candidate pairs the grid contains.
    """
    tri_sorted, pair_off, total_pairs = buildCellIncidences(lo, hi, cell_size)
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
        pr = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], dim=1)
        pr = torch.unique(pr, dim=0)
        yield pr


def buildAABBOverlapPairs(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    inflate: float = 0.0,
    cell_size: Union[float, None] = None,
    pair_chunk: int = 40_000_000,
) -> torch.Tensor:
    """Convenience: all AABB-overlapping triangle pairs (a < b), de-duplicated.

    Suitable when the candidate set is known to be small (e.g. scoped to a moved
    active region). For full-mesh scans prefer `streamOverlapPairs` so peak
    memory stays bounded.
    """
    lo, hi = triangleAABBs(vertices, faces, inflate)
    if cell_size is None:
        cell_size = estimateCellSize(lo, hi, quantile=0.9, factor=1.0)
        if inflate > 0:
            cell_size = max(cell_size, 2.0 * inflate)
    chunks = list(streamOverlapPairs(lo, hi, cell_size, pair_chunk=pair_chunk))
    if not chunks:
        return torch.zeros(0, 2, dtype=torch.long, device=faces.device)
    return torch.unique(torch.cat(chunks, dim=0), dim=0)
