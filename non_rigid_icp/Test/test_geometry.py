"""Unit tests for the topology / geometry / collision / subdivision atoms.

Runnable directly:
    python -m non_rigid_icp.Test.test_geometry
"""

import torch
import numpy as np

from non_rigid_icp.Method.geometry import (
    pointTriangleDistance2,
    segmentSegmentDistance2,
    triangleTriangleDistance2,
    triangleTriangleIntersects,
)
from non_rigid_icp.Method.topology import (
    buildUniqueEdges,
    buildFaceAdjacency,
    buildVertexToFaces,
    dilateFaceMask,
    facePairsShareVertex,
    connectedFaceComponents,
)
from non_rigid_icp.Method.subdivision import subdivideMarkedFaces
from non_rigid_icp.Method.collision import buildCollisionCandidates, detectIntersectingPairs


def _t(x):
    return torch.tensor(x, dtype=torch.float32)


def test_point_triangle_distance():
    a = _t([[0, 0, 0]])
    b = _t([[1, 0, 0]])
    c = _t([[0, 1, 0]])
    # point straight above the interior
    p = _t([[0.25, 0.25, 2.0]])
    d2 = pointTriangleDistance2(p, a, b, c)
    assert abs(d2.item() - 4.0) < 1e-5, d2
    # point coincident with vertex a
    p = _t([[0, 0, 0]])
    d2 = pointTriangleDistance2(p, a, b, c)
    assert d2.item() < 1e-6, d2
    # point off the AB edge
    p = _t([[0.5, -1.0, 0.0]])
    d2 = pointTriangleDistance2(p, a, b, c)
    assert abs(d2.item() - 1.0) < 1e-5, d2
    print("ok  point_triangle_distance")


def test_segment_segment_distance():
    # crossing segments in z=0 -> distance 0
    p1 = _t([[-1, 0, 0]]); q1 = _t([[1, 0, 0]])
    p2 = _t([[0, -1, 0]]); q2 = _t([[0, 1, 0]])
    d2 = segmentSegmentDistance2(p1, q1, p2, q2)
    assert d2.item() < 1e-6, d2
    # parallel offset by 2 in z
    p2 = _t([[0, -1, 2]]); q2 = _t([[0, 1, 2]])
    d2 = segmentSegmentDistance2(p1, q1, p2, q2)
    assert abs(d2.item() - 4.0) < 1e-5, d2
    print("ok  segment_segment_distance")


def test_triangle_triangle_intersect():
    A = _t([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]])  # (1,3,3) in z=0
    # vertical triangle whose edge pierces the interior of A
    B = _t([[[0, 0, -1], [0, 0, 1], [0.5, 0.5, 1]]])
    assert bool(triangleTriangleIntersects(A, B)[0]), "edge-piercing should intersect"
    # separated far above
    B2 = _t([[[0, 0, 5], [0, 0, 7], [0.5, 0.5, 6]]])
    assert not bool(triangleTriangleIntersects(A, B2)[0]), "separated should not intersect"
    # coplanar overlapping
    B3 = _t([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]])
    assert bool(triangleTriangleIntersects(A, B3)[0]), "coplanar identical should intersect"
    # coplanar separated
    B4 = _t([[[5, 5, 0], [7, 5, 0], [6, 7, 0]]])
    assert not bool(triangleTriangleIntersects(A, B4)[0]), "coplanar far should not intersect"
    # near but not touching (gap 0.1) -> no intersect, positive distance
    B5 = _t([[[0, 0, 0.1], [0, 0, 1.1], [0.5, 0.5, 1.1]]])
    assert not bool(triangleTriangleIntersects(A, B5)[0])
    d2 = triangleTriangleDistance2(A, B5)
    assert abs(d2.item() - 0.01) < 1e-4, d2
    print("ok  triangle_triangle_intersect / distance")


def test_topology_quad():
    # two triangles sharing edge (0,2)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    edges = buildUniqueEdges(faces)
    assert edges.shape[0] == 5, edges  # 0-1,1-2,0-2,2-3,0-3
    adj = buildFaceAdjacency(faces)
    assert adj.shape[0] == 1, adj  # one shared edge
    s = set(tuple(sorted(p)) for p in adj.tolist())
    assert (0, 1) in s
    # vertex->faces
    offsets, indices = buildVertexToFaces(faces, 4)
    # vertex 0 in both faces
    f0 = set(indices[offsets[0]:offsets[1]].tolist())
    assert f0 == {0, 1}, f0
    # vertex 1 only in face 0
    f1 = set(indices[offsets[1]:offsets[2]].tolist())
    assert f1 == {0}, f1
    print("ok  topology_quad")


def test_share_vertex_and_dilate():
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=torch.long)
    pairs = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    shared = facePairsShareVertex(pairs, faces)
    assert shared.tolist() == [True, False], shared
    adj = buildFaceAdjacency(faces)
    mask = torch.tensor([True, False, False])
    grown = dilateFaceMask(mask, adj, 1)
    assert grown.tolist() == [True, True, False], grown
    print("ok  share_vertex / dilate")


def test_connected_components():
    # faces 0-1 connected, face 2 separate component, all masked
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=torch.long)
    adj = buildFaceAdjacency(faces)
    mask = torch.tensor([True, True, True])
    labels, sizes = connectedFaceComponents(mask, adj)
    assert sizes.size == 2, sizes
    assert labels[0] == labels[1], labels
    assert labels[2] != labels[0], labels
    print("ok  connected_components")


def _is_watertight(vertices, faces):
    import trimesh
    m = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    return m.is_watertight, m.is_winding_consistent, m.euler_number


def test_subdivision_conforming():
    # tetrahedron (closed, watertight)
    V = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32
    )
    F = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.long
    )
    wt, wind, euler = _is_watertight(V, F)
    assert wt and wind and euler == 2, (wt, wind, euler)

    # subdivide a single face -> must remain watertight & winding-consistent
    region = torch.tensor([True, False, False, False])
    nv, nf, parents = subdivideMarkedFaces(V, F, region)
    # face 0 -> 4 children, its 3 neighbours each -> 2 children
    assert nf.shape[0] == 4 + 3 * 2, nf.shape
    assert nv.shape[0] == 4 + 3, nv.shape  # 3 new midpoints
    wt, wind, euler = _is_watertight(nv, nf)
    assert wt, "subdivided mesh not watertight (cracks/T-junctions!)"
    assert wind, "winding not consistent after subdivision"
    assert euler == 2, euler

    # subdivide all faces (regular 1->4) -> 16 faces, watertight
    region_all = torch.tensor([True, True, True, True])
    nv2, nf2, _ = subdivideMarkedFaces(V, F, region_all)
    assert nf2.shape[0] == 16, nf2.shape
    assert nv2.shape[0] == 4 + 6, nv2.shape  # 6 edge midpoints
    wt, wind, euler = _is_watertight(nv2, nf2)
    assert wt and wind and euler == 2, (wt, wind, euler)
    print("ok  subdivision_conforming")


def test_collision_candidates_on_fold():
    # two parallel sheets very close -> opposing faces are collision candidates,
    # in-sheet adjacent faces must be filtered out.
    V = torch.tensor(
        [
            [0, 0, 0.0], [1, 0, 0.0], [0, 1, 0.0], [1, 1, 0.0],  # sheet z=0
            [0, 0, 0.02], [1, 0, 0.02], [0, 1, 0.02], [1, 1, 0.02],  # sheet z=0.02
        ],
        dtype=torch.float32,
    )
    F = torch.tensor(
        [
            [0, 1, 2], [1, 3, 2],     # bottom sheet
            [4, 5, 6], [5, 7, 6],     # top sheet
        ],
        dtype=torch.long,
    )
    pairs = buildCollisionCandidates(V, F, k=4, margin=0.05, device="cpu")
    # all surviving candidate pairs must be cross-sheet (non-adjacent)
    assert pairs.shape[0] > 0, "should find cross-sheet candidates"
    for i, j in pairs.tolist():
        assert not (i < 2 and j < 2), (i, j)
        assert not (i >= 2 and j >= 2), (i, j)
    # the two sheets do not intersect (gap 0.02) -> no actual intersections
    hit = detectIntersectingPairs(V, F, pairs)
    assert hit.sum().item() == 0, "flat parallel sheets should not intersect"

    # now tilt the top sheet so it crosses the bottom plane -> intersections
    V2 = V.clone()
    V2[4, 2] = -0.01
    V2[5, 2] = 0.01
    V2[6, 2] = -0.01
    V2[7, 2] = 0.01
    hit2 = detectIntersectingPairs(V2, F, pairs)
    assert hit2.sum().item() > 0, "interpenetrating sheets should intersect"
    print("ok  collision_candidates_on_fold")


def main():
    test_point_triangle_distance()
    test_segment_segment_distance()
    test_triangle_triangle_intersect()
    test_topology_quad()
    test_share_vertex_and_dilate()
    test_connected_components()
    test_subdivision_conforming()
    test_collision_candidates_on_fold()
    print("\nALL GEOMETRY/TOPOLOGY/SUBDIVISION TESTS PASSED")


if __name__ == "__main__":
    main()
