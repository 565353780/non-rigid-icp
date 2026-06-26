"""Authoritative global self-intersection scan.

This is independent of the fitter and is the single source of truth for "is this
mesh self-intersection free?". It is used both as an acceptance gate before
saving and as the validation metric.

Pipeline (fully streamed so peak memory is bounded even when dense / already
self-intersecting regions produce billions of within-cell candidate pairs):
  1. inflated triangle AABBs -> grid-hash broad phase (complete: never misses an
     overlapping AABB pair, unlike centroid k-NN);
  2. cheap real-AABB overlap prune;
  3. drop topologically-adjacent pairs (share a vertex, or within a small face
     graph ring) so legitimate seam/fold touches are not counted;
  4. exact triangle-triangle narrow phase (Moller) on the survivors.
"""

import torch
import numpy as np
from typing import Union, Dict, List

from non_rigid_icp.Method.spatial_hash import (
    triangleAABBs,
    estimateCellSize,
    buildCellIncidences,
    cellPairChunk,
    aabbOverlap,
)
from non_rigid_icp.Method.collision import detectIntersectingPairs
from non_rigid_icp.Method.topology import (
    facePairsShareVertex,
    buildFaceAdjacency,
    connectedFaceComponents,
)


def _graph_distance_filter(
    pairs: torch.Tensor,
    faces: torch.Tensor,
    exclude_ring: int,
    face_adjacency: Union[torch.Tensor, None],
) -> torch.Tensor:
    """Drop pairs whose two faces are within `exclude_ring` edge hops.

    Only intended for the (small) post-narrow-phase hit set, so a python BFS per
    pair is acceptable. ring<=1 is handled vectorized elsewhere.
    """
    if pairs.numel() == 0 or exclude_ring <= 1:
        return pairs
    if face_adjacency is None:
        face_adjacency = buildFaceAdjacency(faces)
    from collections import defaultdict

    adj_np = face_adjacency.detach().cpu().numpy()
    nbr = defaultdict(list)
    for u, v in adj_np:
        nbr[int(u)].append(int(v))
        nbr[int(v)].append(int(u))
    a_list = pairs[:, 0].tolist()
    b_list = pairs[:, 1].tolist()
    drop = np.zeros(pairs.shape[0], dtype=bool)
    for i, (fa, fb) in enumerate(zip(a_list, b_list)):
        seen = {fa}
        frontier = {fa}
        for _ in range(exclude_ring):
            nxt = set()
            for u in frontier:
                nxt.update(nbr[u])
            nxt -= seen
            seen |= nxt
            frontier = nxt
            if fb in seen:
                drop[i] = True
                break
    keep = torch.from_numpy(~drop).to(pairs.device)
    return pairs[keep]


def findSelfIntersections(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    inflate: float = 0.0,
    cell_size: Union[float, None] = None,
    exclude_ring: int = 1,
    face_adjacency: Union[torch.Tensor, None] = None,
    query_mask: Union[torch.Tensor, None] = None,
    restrict_to_query: bool = False,
    pair_chunk: int = 40_000_000,
    narrow_chunk: int = 4_000_000,
) -> torch.Tensor:
    """Return (Q, 2) face pairs that actually intersect (non-adjacent).

    Args:
        inflate: AABB inflation. 0 for exact intersection; >0 to also surface
            near-touching pairs (useful when feeding a barrier).
        exclude_ring: topological adjacency exclusion. 1 = drop shared-vertex
            pairs (standard); >=2 also drops within-ring graph neighbours.
        query_mask: optional (F,) bool; keep only pairs with >=1 endpoint in it.
        restrict_to_query: when True (and `query_mask` given), build the grid
            incidences over ONLY the query faces, so the broad phase costs scale
            with the (small) active region rather than the full mesh. This finds
            query-vs-query crossings only (the in-loop fast gate: a collapsing
            double layer moves both its sheets, so both are in the active query);
            the rare query-vs-static crossing is left for the full-mesh gate.
            When False the grid spans all faces and `query_mask` only filters the
            generated pairs (authoritative / complete).
    """
    device = faces.device
    if restrict_to_query and query_mask is not None and bool(query_mask.any()):
        q_ids = torch.nonzero(query_mask, as_tuple=False).reshape(-1)
        index_faces = faces[q_ids]
        local_to_global = q_ids
        post_query = None  # every sub-pair already touches the active region
    else:
        index_faces = faces
        local_to_global = None
        post_query = query_mask

    lo, hi = triangleAABBs(vertices, index_faces, inflate)
    if cell_size is None:
        cell_size = estimateCellSize(lo, hi, quantile=0.9, factor=1.0)
        if inflate > 0:
            cell_size = max(cell_size, 2.0 * inflate)

    tri_sorted, pair_off, total_pairs = buildCellIncidences(lo, hi, cell_size)
    hits: List[torch.Tensor] = []
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
        if post_query is not None:
            sel = post_query[pr[:, 0]] | post_query[pr[:, 1]]
            pr = pr[sel]
            if pr.numel() == 0:
                continue
        # adjacency + narrow phase run in the same index space as `pr`
        shared = facePairsShareVertex(pr, index_faces)
        pr = pr[~shared]
        if pr.numel() == 0:
            continue
        hit = detectIntersectingPairs(vertices, index_faces, pr, chunk=narrow_chunk)
        if bool(hit.any()):
            hits.append(pr[hit])

    if not hits:
        return torch.zeros(0, 2, dtype=torch.long, device=device)
    inter = torch.unique(torch.cat(hits, dim=0), dim=0)
    if local_to_global is not None:
        inter = torch.sort(local_to_global[inter], dim=1).values
    inter = _graph_distance_filter(inter, faces, exclude_ring, face_adjacency)
    return inter


def selfIntersectionReport(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    inflate: float = 0.0,
    cell_size: Union[float, None] = None,
    exclude_ring: int = 1,
    cluster: bool = True,
) -> Dict:
    """Full diagnostic report for a mesh.

    Returns a dict with intersecting pair count, number of distinct faces
    involved, fraction of faces, and (optionally) connected-component clustering
    of the intersecting faces (region count + largest region size).
    """
    inter = findSelfIntersections(
        vertices, faces, inflate=inflate, cell_size=cell_size, exclude_ring=exclude_ring
    )
    f = faces.shape[0]
    report: Dict = {
        "num_faces": int(f),
        "num_intersecting_pairs": int(inter.shape[0]),
    }
    if inter.shape[0] == 0:
        report.update(
            {
                "num_intersecting_faces": 0,
                "frac_intersecting_faces": 0.0,
                "num_regions": 0,
                "largest_region_faces": 0,
            }
        )
        return report

    involved = torch.unique(inter.reshape(-1))
    report["num_intersecting_faces"] = int(involved.numel())
    report["frac_intersecting_faces"] = float(involved.numel()) / float(f)

    if cluster:
        face_mask = torch.zeros(f, dtype=torch.bool, device=faces.device)
        face_mask[involved] = True
        face_adj = buildFaceAdjacency(faces)
        labels, sizes = connectedFaceComponents(face_mask, face_adj)
        report["num_regions"] = int((sizes > 0).sum())
        report["largest_region_faces"] = int(sizes.max()) if sizes.size else 0
    return report
