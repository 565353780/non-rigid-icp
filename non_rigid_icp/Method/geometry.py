"""Vectorized geometric distance and intersection primitives.

All functions are batched over the leading dimension and differentiable where it
makes sense (the distance kernels), so they can serve both the hard
self-intersection guard and the soft (barrier) collision loss. They follow
Ericson, "Real-Time Collision Detection" for the distance kernels and Moller,
"A Fast Triangle-Triangle Intersection Test" for the boolean intersection test.
"""

import torch
from typing import Tuple

EPS = 1e-12


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


def closestPointOnSegment(
    p: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Closest point to p on segment [a, b]. All (..., 3). Returns (..., 3)."""
    ab = b - a
    t = _dot(p - a, ab) / (_dot(ab, ab) + EPS)
    t = t.clamp(0.0, 1.0).unsqueeze(-1)
    return a + t * ab


def pointTriangleDistance2(
    p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Squared distance from points p to triangles (a, b, c).

    All inputs (..., 3). Returns (...,). Vectorized Ericson closest-point logic;
    the seven Voronoi regions are mutually exclusive so we select by masks.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)

    bp = p - b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)

    cp = p - c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # default: project onto the face interior
    denom = 1.0 / (va + vb + vc + EPS)
    v_face = (vb * denom).unsqueeze(-1)
    w_face = (vc * denom).unsqueeze(-1)
    closest = a + ab * v_face + ac * w_face

    # edge BC
    bc_w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + EPS)
    closest_bc = b + bc_w.unsqueeze(-1) * (c - b)
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    closest = torch.where(mask_bc.unsqueeze(-1), closest_bc, closest)

    # edge AC
    ac_w = (d2 / (d2 - d6 + EPS)).unsqueeze(-1)
    closest_ac = a + ac_w * ac
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    closest = torch.where(mask_ac.unsqueeze(-1), closest_ac, closest)

    # vertex C
    mask_c = (d6 >= 0) & (d5 <= d6)
    closest = torch.where(mask_c.unsqueeze(-1), c, closest)

    # edge AB
    ab_v = (d1 / (d1 - d3 + EPS)).unsqueeze(-1)
    closest_ab = a + ab_v * ab
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    closest = torch.where(mask_ab.unsqueeze(-1), closest_ab, closest)

    # vertex B
    mask_b = (d3 >= 0) & (d4 <= d3)
    closest = torch.where(mask_b.unsqueeze(-1), b, closest)

    # vertex A
    mask_a = (d1 <= 0) & (d2 <= 0)
    closest = torch.where(mask_a.unsqueeze(-1), a, closest)

    diff = p - closest
    return _dot(diff, diff)


def segmentSegmentDistance2(
    p1: torch.Tensor, q1: torch.Tensor, p2: torch.Tensor, q2: torch.Tensor
) -> torch.Tensor:
    """Squared distance between segments [p1,q1] and [p2,q2]. All (..., 3)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, r)
    c = _dot(d1, r)
    b = _dot(d1, d2)
    denom = a * e - b * b

    s = torch.where(denom > EPS, (b * f - c * e) / (denom + EPS), torch.zeros_like(denom))
    s = s.clamp(0.0, 1.0)

    t = (b * s + f) / (e + EPS)
    t_clamped = t.clamp(0.0, 1.0)
    s = ((b * t_clamped - c) / (a + EPS)).clamp(0.0, 1.0)

    closest1 = p1 + d1 * s.unsqueeze(-1)
    closest2 = p2 + d2 * t_clamped.unsqueeze(-1)
    diff = closest1 - closest2
    return _dot(diff, diff)


def triangleTriangleDistance2(tri1: torch.Tensor, tri2: torch.Tensor) -> torch.Tensor:
    """Squared distance between disjoint triangle pairs.

    Args:
        tri1: (P, 3, 3) vertices of the first triangles.
        tri2: (P, 3, 3) vertices of the second triangles.

    Returns:
        (P,) squared distance, computed as the min over 6 vertex-triangle and 9
        edge-edge sub-distances. For triangles that do NOT cross this is the true
        separation; for crossing triangles it can be > 0, so it is a *preventive*
        proxy and the boolean test below is used to detect actual crossings.
    """
    a1, b1, c1 = tri1[:, 0], tri1[:, 1], tri1[:, 2]
    a2, b2, c2 = tri2[:, 0], tri2[:, 1], tri2[:, 2]

    # 6 vertex-triangle distances
    d = pointTriangleDistance2(a1, a2, b2, c2)
    d = torch.minimum(d, pointTriangleDistance2(b1, a2, b2, c2))
    d = torch.minimum(d, pointTriangleDistance2(c1, a2, b2, c2))
    d = torch.minimum(d, pointTriangleDistance2(a2, a1, b1, c1))
    d = torch.minimum(d, pointTriangleDistance2(b2, a1, b1, c1))
    d = torch.minimum(d, pointTriangleDistance2(c2, a1, b1, c1))

    # 9 edge-edge distances
    edges1 = [(a1, b1), (b1, c1), (c1, a1)]
    edges2 = [(a2, b2), (b2, c2), (c2, a2)]
    for p1, q1 in edges1:
        for p2, q2 in edges2:
            d = torch.minimum(d, segmentSegmentDistance2(p1, q1, p2, q2))
    return d


def _signed_distances_to_plane(
    tri_pts: torch.Tensor, normal: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    # tri_pts: (P, 3, 3); normal: (P, 3); d: (P,)
    return (tri_pts * normal.unsqueeze(1)).sum(dim=-1) + d.unsqueeze(1)


def _crossing_interval(
    p: torch.Tensor, dist: torch.Tensor, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interval where a triangle crosses a plane, on the intersection-line param.

    p: (P, 3) projected vertex coordinates, dist: (P, 3) signed plane distances.
    Returns (t_min, t_max) over the crossing points. Two contributions are
    combined: edges whose endpoints straddle the plane (sign change) and
    vertices lying exactly on the plane (|dist| <= eps). Including the on-plane
    vertices is essential, otherwise a triangle touching the plane at a vertex
    loses an interval endpoint and intersections are missed.
    """
    i = [0, 1, 2]
    j = [1, 2, 0]
    di = dist[:, i]
    dj = dist[:, j]
    pi = p[:, i]
    pj = p[:, j]

    cross = (di * dj) < 0
    denom = di - dj
    safe_denom = torch.where(denom.abs() > eps, denom, torch.ones_like(denom))
    t_edge = pi + (pj - pi) * di / safe_denom

    on_plane = dist.abs() <= eps  # (P, 3)

    pos_inf_e = torch.full_like(t_edge, float("inf"))
    neg_inf_e = torch.full_like(t_edge, float("-inf"))
    pos_inf_v = torch.full_like(p, float("inf"))
    neg_inf_v = torch.full_like(p, float("-inf"))

    t_min_edge = torch.where(cross, t_edge, pos_inf_e).min(dim=1).values
    t_max_edge = torch.where(cross, t_edge, neg_inf_e).max(dim=1).values
    t_min_vert = torch.where(on_plane, p, pos_inf_v).min(dim=1).values
    t_max_vert = torch.where(on_plane, p, neg_inf_v).max(dim=1).values

    t_min = torch.minimum(t_min_edge, t_min_vert)
    t_max = torch.maximum(t_max_edge, t_max_vert)
    return t_min, t_max


def triangleTriangleIntersects(
    tri1: torch.Tensor, tri2: torch.Tensor, coplanar_eps: float = 1e-9
) -> torch.Tensor:
    """Boolean test whether triangle pairs intersect (Moller's method).

    Args:
        tri1: (P, 3, 3), tri2: (P, 3, 3).

    Returns:
        (P,) bool. Non-coplanar pairs are handled exactly; (near-)coplanar pairs
        fall back to a conservative touching test (distance ~ 0), which combined
        with the barrier loss is sufficient for the deformation guard.
    """
    if tri1.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=tri1.device)

    a1, b1, c1 = tri1[:, 0], tri1[:, 1], tri1[:, 2]
    a2, b2, c2 = tri2[:, 0], tri2[:, 1], tri2[:, 2]

    n1 = torch.cross(b1 - a1, c1 - a1, dim=-1)
    d1 = -_dot(n1, a1)
    n2 = torch.cross(b2 - a2, c2 - a2, dim=-1)
    d2 = -_dot(n2, a2)

    # signed distances of each triangle's vertices to the other plane
    dist_1to2 = _signed_distances_to_plane(tri1, n2, d2)  # (P, 3)
    dist_2to1 = _signed_distances_to_plane(tri2, n1, d1)  # (P, 3)

    # if all three on the same (strict) side, no intersection
    same_side_1 = (dist_1to2 > 0).all(dim=1) | (dist_1to2 < 0).all(dim=1)
    same_side_2 = (dist_2to1 > 0).all(dim=1) | (dist_2to1 < 0).all(dim=1)
    separated = same_side_1 | same_side_2

    # intersection line direction
    dir_line = torch.cross(n1, n2, dim=-1)
    coplanar = _dot(dir_line, dir_line) < coplanar_eps

    # project vertices onto the intersection line direction
    p1 = (tri1 * dir_line.unsqueeze(1)).sum(dim=-1)  # (P, 3)
    p2 = (tri2 * dir_line.unsqueeze(1)).sum(dim=-1)

    amin, amax = _crossing_interval(p1, dist_1to2)
    bmin, bmax = _crossing_interval(p2, dist_2to1)
    overlap = (torch.maximum(amin, bmin) <= torch.minimum(amax, bmax))

    noncoplanar_hit = overlap & (~separated) & (~coplanar)

    # conservative coplanar fallback: treat as intersecting if effectively touching
    dist2 = triangleTriangleDistance2(tri1, tri2)
    coplanar_hit = coplanar & (dist2 <= coplanar_eps)

    return noncoplanar_hit | coplanar_hit
