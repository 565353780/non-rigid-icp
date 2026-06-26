"""Unit + integration tests for the implicit-field projection fitter.

    python -m non_rigid_icp.Test.test_projection_fitter

Covers the new atomic functions (segment-triangle intersection, trajectory
guard, wall thickness, rest-field carry through subdivision) and the end-to-end
double-layer collapse -- the exact failure mode the projection guard must defeat
with zero self-intersections.
"""

import numpy as np
import torch
import open3d as o3d

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.projection_fitter import ProjectionFitter
from non_rigid_icp.Method.self_intersection import findSelfIntersections
from non_rigid_icp.Method.geometry import segmentTriangleIntersect
from non_rigid_icp.Method.trajectory_guard import segmentsCrossMesh, largestSafeStep
from non_rigid_icp.Method.thickness import (
    vertexWallThickness,
    vertexWallPartner,
    inwardComponentCap,
)
from non_rigid_icp.Method.subdivision import subdivideMarkedFaces


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_segment_triangle():
    a = torch.tensor([[0.0, 0.0, 0.0]])
    b = torch.tensor([[1.0, 0.0, 0.0]])
    c = torch.tensor([[0.0, 1.0, 0.0]])

    # pierces the interior
    p = torch.tensor([[0.25, 0.25, -0.5]])
    q = torch.tensor([[0.25, 0.25, 0.5]])
    assert bool(segmentTriangleIntersect(p, q, a, b, c)[0])

    # passes outside the triangle (u + v > 1)
    p = torch.tensor([[0.8, 0.8, -0.5]])
    q = torch.tensor([[0.8, 0.8, 0.5]])
    assert not bool(segmentTriangleIntersect(p, q, a, b, c)[0])

    # does not reach the plane (segment entirely above)
    p = torch.tensor([[0.25, 0.25, 0.5]])
    q = torch.tensor([[0.25, 0.25, 1.5]])
    assert not bool(segmentTriangleIntersect(p, q, a, b, c)[0])

    # coplanar segment does not "pierce"
    p = torch.tensor([[-1.0, 0.25, 0.0]])
    q = torch.tensor([[2.0, 0.25, 0.0]])
    assert not bool(segmentTriangleIntersect(p, q, a, b, c)[0])
    print("  test_segment_triangle PASSED")


def test_trajectory_guard():
    # a single horizontal sheet (z = 0) made of two triangles, plus one free
    # vertex that wants to move from far above to far below: its trajectory must
    # pierce the sheet and be clamped back to just above it.
    verts = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],  # the mover (owner vertex id 4), sits above the sheet
        ],
        device=DEV,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], device=DEV, dtype=torch.long)

    ref = verts[4:5].clone()  # rest just above the sheet
    proposed = torch.tensor([[0.0, 0.0, -1.0]], device=DEV)  # wants to tunnel through
    owner = torch.tensor([4], device=DEV, dtype=torch.long)

    crossed = segmentsCrossMesh(ref, proposed, verts, faces, owner)
    assert bool(crossed[0]), "trajectory through the sheet must be detected"

    safe, clamped = largestSafeStep(
        ref, proposed, verts, faces, owner_vid=owner, n_bisect=12
    )
    assert bool(clamped[0])
    assert float(safe[0, 2]) > 0.0, "safe position must stay on the rest side (z>0)"

    # a vertex incident to the sheet moving tangentially must NOT be flagged
    owner_inc = torch.tensor([0], device=DEV, dtype=torch.long)
    ref_inc = verts[0:1].clone()
    prop_inc = verts[0:1] + torch.tensor([[0.1, 0.1, 0.0]], device=DEV)
    crossed_inc = segmentsCrossMesh(ref_inc, prop_inc, verts, faces, owner_inc)
    assert not bool(crossed_inc[0]), "incident-face moves must be excluded"
    print("  test_trajectory_guard PASSED")


def test_wall_thickness():
    g = 0.05
    base = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    v0 = base.copy()
    v1 = base.copy()
    v1[:, 2] = g
    verts = np.concatenate([v0, v1], axis=0)
    f0 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    f1 = f0 + 4
    faces = np.concatenate([f0, f1], axis=0)
    vt = vertexWallThickness(
        torch.tensor(verts, device=DEV),
        torch.tensor(faces, device=DEV),
        max_thickness=10.0 * g,
        eps=0.1 * g,  # the planes are unit-sized but the gap is tiny; offset << gap
    )
    finite = vt[torch.isfinite(vt)]
    assert finite.numel() > 0
    assert abs(float(finite.median()) - g) < 0.2 * g, float(finite.median())
    print("  test_wall_thickness PASSED")


def test_inward_cap():
    # two parallel sheets a gap `g` apart along z; the partner direction of the
    # bottom sheet must point +z (toward the top) and vice versa.
    g = 0.5
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    v0 = base.copy()
    v1 = base.copy()
    v1[:, 2] = g
    verts = torch.tensor(np.concatenate([v0, v1], axis=0), device=DEV)
    f0 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    faces = torch.tensor(np.concatenate([f0, f0 + 4], axis=0), device=DEV)

    thickness, toward = vertexWallPartner(verts, faces, max_thickness=10.0 * g)
    assert torch.isfinite(thickness).all(), "both sheets should see a partner"
    assert abs(float(thickness.median()) - g) < 0.2 * g
    # bottom verts (0..3) point up, top verts (4..7) point down
    assert (toward[:4, 2] > 0.5).all(), toward[:4]
    assert (toward[4:, 2] < -0.5).all(), toward[4:]

    # margin 0.4 -> allowance per layer = (0.5-0.4)/2 = 0.05; a 0.3 inward step
    # must be clamped so each layer moved at most 0.05 toward the partner.
    ref = verts.clone()
    margin = 0.4
    allowance = torch.clamp((thickness - margin) * 0.5, min=0.0)
    collapsed = verts.clone()
    collapsed[:4, 2] += 0.3  # bottom shoved up (toward partner)
    collapsed[4:, 2] -= 0.3  # top shoved down (toward partner)
    fixed = inwardComponentCap(collapsed, ref, toward, allowance)
    gap = fixed[4:, 2].mean() - fixed[:4, 2].mean()
    assert float(gap) >= margin - 1e-4, float(gap)

    # OUTWARD / translation motion must pass through untouched
    out = verts.clone()
    out[:4, 2] -= 0.3  # bottom moves down (away from partner) -> allowed
    keep = inwardComponentCap(out, ref, toward, allowance)
    assert torch.allclose(keep, out, atol=1e-5), "outward motion was altered"
    print("  test_inward_cap PASSED")


def test_subdivision_rest_carry():
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=DEV
    )
    faces = torch.tensor([[0, 1, 2]], device=DEV, dtype=torch.long)
    # distinct rest positions (shifted), so midpoints are easy to verify
    ref = verts + torch.tensor([10.0, 20.0, 30.0], device=DEV)
    mask = torch.ones(1, dtype=torch.bool, device=DEV)
    new_v, new_f, _, extra = subdivideMarkedFaces(
        verts, faces, mask, extra_vertex_attrs=[ref]
    )
    new_ref = extra[0]
    # every new midpoint's rest = its position + the same constant shift, i.e.
    # the interpolated point on the watertight mesh edge.
    shift = torch.tensor([10.0, 20.0, 30.0], device=DEV)
    assert torch.allclose(new_ref, new_v + shift, atol=1e-5)
    print("  test_subdivision_rest_carry PASSED")


def _sphere(radius, resolution=24, flip=False):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    if flip:
        t = np.asarray(m.triangles)[:, ::-1].copy()
        m.triangles = o3d.utility.Vector3iVector(t)
    return m


def _shell(r_in, r_out, resolution=24):
    outer = _sphere(r_out, resolution)
    inner = _sphere(r_in, resolution, flip=True)
    ov, ot = np.asarray(outer.vertices), np.asarray(outer.triangles)
    iv = np.asarray(inner.vertices)
    it = np.asarray(inner.triangles) + ov.shape[0]
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.concatenate([ov, iv], axis=0))
    m.triangles = o3d.utility.Vector3iVector(np.concatenate([ot, it], axis=0))
    m.compute_vertex_normals()
    return m


def _run_shell(protect):
    source = Mesh.from_o3d(_shell(r_in=0.447, r_out=0.453, resolution=30))
    target = Mesh.from_o3d(_sphere(0.450, resolution=40))
    fitter = ProjectionFitter(
        device=DEV,
        step_tau=1.0,
        max_sweeps=24,
        # PROTECT: detect the wall (max_thickness 20 tau) and cap each layer's
        #   cumulative motion toward its partner to keep a >=0.5 tau gap, plus the
        #   trajectory guard and the strict final gate -> the shell stays open.
        # UNPROTECTED: disable thickness detection (max_thickness 0 -> no partner,
        #   no cap) and the guard so both layers re-project onto the mid sphere
        #   and coincide.
        max_thickness_tau=(20.0 if protect else 0.0),
        gap_margin_tau=0.5,
        enable_guard=protect,
        strict_no_intersection=protect,
        max_subdivisions=0,
        train_target_samples=150000,
        eval_samples=150000,
        save_result_folder_path=None,
        seed=0,
    )
    fitter.loadMeshes(source, target)
    result = fitter.fitAndEvaluate()
    V = torch.tensor(
        np.asarray(fitter.source_mesh.vertices, dtype=np.float32), device=DEV
    )
    F = torch.tensor(
        np.asarray(fitter.source_mesh.triangles, dtype=np.int64), device=DEV
    )
    inter = findSelfIntersections(V, F, inflate=0.0, exclude_ring=1)
    return result, int(inter.shape[0])


def test_double_layer():
    print("  === UNPROTECTED (no cap, no guard) ===")
    bare, bare_si = _run_shell(protect=False)
    print("    chamfer_l1:", round(bare["fitted"]["chamfer_l1"], 6))
    print("    independent self-intersections:", bare_si)

    print("  === PROTECTED (gap cap + guard + final gate) ===")
    prot, prot_si = _run_shell(protect=True)
    print("    chamfer_l1:", round(prot["fitted"]["chamfer_l1"], 6))
    print("    kept:", prot["kept"])
    print("    reported new self-intersections:", prot["final_new_self_intersections"])
    print("    independent self-intersections:", prot_si)

    assert bare_si > 50, (
        f"expected the unprotected shell collapse to self-intersect; got {bare_si}"
    )
    assert prot_si == 0, (
        f"protection failed to prevent shell collapse: {prot_si} new intersections"
    )
    assert prot["fitted"]["chamfer_l1"] < prot["baseline"]["chamfer_l1"], (
        "protected projection must still improve on the rigid baseline"
    )
    print("  test_double_layer PASSED")


def main():
    print("[test] segment-triangle primitive")
    test_segment_triangle()
    print("[test] trajectory guard")
    test_trajectory_guard()
    print("[test] wall thickness")
    test_wall_thickness()
    print("[test] inward-component cap")
    test_inward_cap()
    print("[test] subdivision rest carry")
    test_subdivision_rest_carry()
    print("[test] double-layer collapse (end to end)")
    test_double_layer()
    print("\nALL PROJECTION FITTER TESTS PASSED")


if __name__ == "__main__":
    main()
