"""Tests for the v4 real-time trajectory self-intersection guard.

    python -m non_rigid_icp.Test.test_v4_trajectory_guard

Covers the new detailed atoms (segmentMeshIntersections, largestSafeStep hit
reporting, local face scope) and the end-to-end WatertightFitter integration:
the user-defined criterion -- a vertex's straight trajectory from its watertight
rest to its fitted position must not pierce any non-incident face -- is enforced
in real time by locally dragging offending vertices back (never resetting them
wholesale to rest), and the final output is trajectory-crossing free.
"""

import numpy as np
import torch
import open3d as o3d

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter
from non_rigid_icp.Method.trajectory_guard import (
    segmentMeshIntersections,
    segmentMeshIntersectionParams,
    earliestSegmentMeshHits,
    largestSafeStep,
    earliestSafeStep,
)
from non_rigid_icp.Method.geometry import segmentTriangleIntersectParams
from non_rigid_icp.Method.triton_kernels import (
    segmentTrianglePairHits,
    segmentTrianglePairParams,
    tritonAvailable,
)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _flat_sheet():
    """A single horizontal sheet z=0 made of two triangles, plus a free mover
    vertex (id 4) sitting above it."""
    verts = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-0.2, -0.2, 1.0],  # mover, clearly inside triangle (0,1,2)
        ],
        device=DEV,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], device=DEV, dtype=torch.long)
    return verts, faces


def test_segment_mesh_intersections():
    verts, faces = _flat_sheet()
    ref = verts[4:5].clone()  # z = 1, above
    cur = ref.clone()
    cur[0, 2] = -1.0  # tunnels straight through the sheet
    owner = torch.tensor([4], device=DEV, dtype=torch.long)

    hit_seg, hit_face, crossed = segmentMeshIntersections(ref, cur, verts, faces, owner)
    assert bool(crossed[0]), "trajectory through the sheet must be detected"
    assert hit_seg.numel() >= 1
    # the mover sits inside triangle 0, so face 0 must be among the pierced faces
    assert 0 in set(hit_face.tolist()), hit_face.tolist()

    # an incident move (a sheet corner sliding tangentially) is never a crossing
    owner_inc = torch.tensor([0], device=DEV, dtype=torch.long)
    ref_inc = verts[0:1].clone()
    cur_inc = verts[0:1] + torch.tensor([[0.1, 0.1, 0.0]], device=DEV)
    _, _, crossed_inc = segmentMeshIntersections(
        ref_inc, cur_inc, verts, faces, owner_inc
    )
    assert not bool(crossed_inc[0]), "incident-face moves must be excluded"
    print("  test_segment_mesh_intersections PASSED")


def test_local_face_scope():
    verts, faces = _flat_sheet()
    ref = verts[4:5].clone()
    cur = ref.clone()
    cur[0, 2] = -1.0
    owner = torch.tensor([4], device=DEV, dtype=torch.long)

    # restricting to the face the mover actually pierces (0) still detects it,
    # and the returned face id is remapped back to the GLOBAL id 0.
    _, hit_face0, crossed0 = segmentMeshIntersections(
        ref, cur, verts, faces, owner, face_ids=torch.tensor([0], device=DEV)
    )
    assert bool(crossed0[0]) and 0 in set(hit_face0.tolist())

    # restricting to the OTHER face (1) misses it -- proof the scope is honored.
    _, _, crossed1 = segmentMeshIntersections(
        ref, cur, verts, faces, owner, face_ids=torch.tensor([1], device=DEV)
    )
    assert not bool(crossed1[0]), "scope restricted to face 1 must not detect a hit"
    print("  test_local_face_scope PASSED")


def test_largest_safe_step_hits_and_pullback():
    verts, faces = _flat_sheet()
    ref = verts[4:5].clone()  # z = 1
    proposed = ref.clone()
    proposed[0, 2] = -1.0  # wants to tunnel to z = -1
    owner = torch.tensor([4], device=DEV, dtype=torch.long)

    safe, clamped, hit_seg, hit_face = largestSafeStep(
        ref, proposed, verts, faces, owner_vid=owner, n_bisect=16, return_hits=True
    )
    assert bool(clamped[0]), "the tunneling step must be clamped"
    # pulled back to the largest safe fraction, NOT reset to rest: it stays on the
    # rest side (z>0) but moved well below the rest z=1.
    assert float(safe[0, 2]) > 0.0, "safe position must stay on the rest side"
    assert float(safe[0, 2]) < 1.0, "safe position must have advanced from rest"
    assert hit_seg.numel() >= 1 and 0 in set(hit_face.tolist())
    print("  test_largest_safe_step_hits_and_pullback PASSED")


def test_param_atom_t_value():
    """A vertical segment from z=+1 to z=-1 through the z=0 sheet pierces at the
    geometric midpoint, so the parameter t must be exactly 0.5."""
    p = torch.tensor([[0.0, 0.0, 1.0]], device=DEV)
    q = torch.tensor([[0.0, 0.0, -1.0]], device=DEV)
    a = torch.tensor([[-1.0, -1.0, 0.0]], device=DEV)
    b = torch.tensor([[1.0, -1.0, 0.0]], device=DEV)
    c = torch.tensor([[-1.0, 1.0, 0.0]], device=DEV)
    hit, t, u, v = segmentTriangleIntersectParams(p, q, a, b, c)
    assert bool(hit[0])
    assert abs(float(t[0]) - 0.5) < 1e-5, float(t[0])

    # raising t_lo above the rest endpoint should not affect a midpoint crossing
    hit2, t2, _, _ = segmentTriangleIntersectParams(p, q, a, b, c, t_lo=0.01)
    assert bool(hit2[0]) and abs(float(t2[0]) - 0.5) < 1e-5

    # a segment that only touches the sheet at its REST endpoint (t=0) is a hit
    # under the default window but excluded once t_lo is lifted above 0.
    p0 = torch.tensor([[0.0, 0.0, 0.0]], device=DEV)  # on the sheet
    q0 = torch.tensor([[0.0, 0.0, 1.0]], device=DEV)  # moves away
    hit_def, _, _, _ = segmentTriangleIntersectParams(p0, q0, a, b, c)
    hit_eps, t_eps, _, _ = segmentTriangleIntersectParams(
        p0, q0, a, b, c, t_lo=1e-3
    )
    assert bool(hit_def[0]), "default window keeps the t=0 rest touch"
    assert not bool(hit_eps[0]), "t_lo>0 must drop the rest-endpoint touch"
    assert not torch.isfinite(t_eps[0]), "dropped pairs get t=+inf"
    print("  test_param_atom_t_value PASSED")


def test_earliest_hit_reduction():
    """Two parallel sheets at z=0.25 and z=0.75; a segment from z=1 to z=0 must
    report the EARLIEST crossing (closest to rest, smaller t) -- the z=0.75 sheet
    at t=0.25, not the z=0.25 sheet at t=0.75."""
    verts = torch.tensor(
        [
            [-1.0, -1.0, 0.25], [1.0, -1.0, 0.25], [-1.0, 1.0, 0.25],  # low sheet
            [-1.0, -1.0, 0.75], [1.0, -1.0, 0.75], [-1.0, 1.0, 0.75],  # high sheet
            [0.0, 0.0, 1.0],  # mover (id 6), rest above both sheets
        ],
        device=DEV,
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device=DEV, dtype=torch.long)
    ref = verts[6:7].clone()
    cur = ref.clone()
    cur[0, 2] = 0.0  # tunnels through both sheets
    owner = torch.tensor([6], device=DEV, dtype=torch.long)

    hit_seg, hit_face, hit_t = segmentMeshIntersectionParams(
        ref, cur, verts, faces, owner
    )
    assert hit_seg.numel() == 2, hit_seg.tolist()
    t_min, face_min = earliestSegmentMeshHits(hit_seg, hit_face, hit_t, 1)
    # rest is z=1, current z=0, span 1.0: z=0.75 sheet is at t=0.25 (earliest).
    assert abs(float(t_min[0]) - 0.25) < 1e-5, float(t_min[0])
    assert int(face_min[0]) == 1, "earliest crossing is the HIGH sheet (face 1)"
    print("  test_earliest_hit_reduction PASSED")


def test_earliest_safe_step_min_pullback():
    """earliestSafeStep places the vertex just before its first crossing, in one
    closed-form step (matching largestSafeStep's bisection to clearance)."""
    verts, faces = _flat_sheet()
    ref = verts[4:5].clone()  # z = 1
    proposed = ref.clone()
    proposed[0, 2] = -1.0  # tunnels to z=-1, crossing the z=0 sheet at t=0.5
    owner = torch.tensor([4], device=DEV, dtype=torch.long)

    safe, clamped, hit_seg, hit_face = earliestSafeStep(
        ref, proposed, verts, faces, owner_vid=owner,
        clearance_t=1e-3, return_hits=True,
    )
    assert bool(clamped[0])
    # crossing at t=0.5 (z=0); pulled back to t=0.5-clearance -> z slightly >0.
    z = float(safe[0, 2])
    assert 0.0 < z < 0.01, z
    assert hit_seg.numel() == 1 and int(hit_face[0]) == 0
    print("  test_earliest_safe_step_min_pullback PASSED")


def test_triton_param_matches_torch():
    """The Triton parameter kernel matches the torch fallback on random pairs
    (both the boolean pierce flag and the parameter t where finite)."""
    if not (tritonAvailable() and DEV == "cuda"):
        print("  test_triton_param_matches_torch SKIPPED (no triton/cuda)")
        return
    g = torch.Generator(device=DEV).manual_seed(0)
    s, t_, p = 200, 200, 50000
    ss = torch.rand(s, 3, device=DEV, generator=g)
    se = torch.rand(s, 3, device=DEV, generator=g)
    va = torch.rand(t_, 3, device=DEV, generator=g)
    vb = torch.rand(t_, 3, device=DEV, generator=g)
    vc = torch.rand(t_, 3, device=DEV, generator=g)
    seg_id = torch.randint(0, s, (p,), device=DEV, generator=g)
    tri_id = torch.randint(0, t_, (p,), device=DEV, generator=g)

    t_k = segmentTrianglePairParams(seg_id, tri_id, ss, se, va, vb, vc)
    _, t_ref, _, _ = segmentTriangleIntersectParams(
        ss[seg_id], se[seg_id], va[tri_id], vb[tri_id], vc[tri_id]
    )
    # both agree on which pairs are hits (finite t)
    assert torch.equal(torch.isfinite(t_k), torch.isfinite(t_ref))
    fin = torch.isfinite(t_k)
    if bool(fin.any()):
        assert torch.allclose(t_k[fin], t_ref[fin], atol=1e-4), (
            (t_k[fin] - t_ref[fin]).abs().max().item()
        )
    # the boolean kernel must agree with finiteness of the parameter kernel
    hit_b = segmentTrianglePairHits(seg_id, tri_id, ss, se, va, vb, vc)
    assert torch.equal(hit_b, fin)
    print("  test_triton_param_matches_torch PASSED")


def _sphere(radius: float, resolution: int = 24, flip: bool = False):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    if flip:
        t = np.asarray(m.triangles)[:, ::-1].copy()
        m.triangles = o3d.utility.Vector3iVector(t)
    return m


def _shell(r_in: float, r_out: float, resolution: int = 24):
    outer = _sphere(r_out, resolution)
    inner = _sphere(r_in, resolution, flip=True)
    ov = np.asarray(outer.vertices)
    ot = np.asarray(outer.triangles)
    iv = np.asarray(inner.vertices)
    it = np.asarray(inner.triangles) + ov.shape[0]
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.concatenate([ov, iv], axis=0))
    m.triangles = o3d.utility.Vector3iVector(np.concatenate([ot, it], axis=0))
    m.compute_vertex_normals()
    return m


def _run_shell(enable_trajectory_guard: bool):
    """A thin double-layer shell forced to collapse onto its mid sphere -- with
    the normal gate and every soft barrier OFF, so the only thing that can keep
    the inner layer from tunnelling through the outer is the trajectory guard."""
    source = Mesh.from_o3d(_shell(r_in=0.447, r_out=0.453, resolution=24))
    target = Mesh.from_o3d(_sphere(0.450, resolution=32))
    fitter = WatertightFitter(
        device=DEV,
        outer_iter=120,
        inner_iter=8,
        lr=2e-4,
        train_source_samples=60000,
        train_target_samples=120000,
        eval_samples=120000,
        laplacian_weight=1.0,
        normal_gate=False,
        # isolate the trajectory guard: disable every other guard
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=enable_trajectory_guard,
        trajectory_check_inner_every=2,
        trajectory_active_tau=0.25,
        trajectory_bisect_steps=14,
        trajectory_resolve_iters=6,
        trajectory_final_rounds=40,
        max_subdivisions=0,
        save_result_folder_path=None,
        seed=0,
    )
    fitter.loadMeshes(source, target)
    result = fitter.fitAndEvaluate()

    # independent re-measurement of the user criterion on the de-normalized
    # output against its reference mesh (topology must match).
    out = fitter.source_mesh
    ref_mesh = fitter._trajectory_reference_mesh
    cur = torch.tensor(np.asarray(out.vertices, dtype=np.float32), device=DEV)
    faces = torch.tensor(np.asarray(out.triangles, dtype=np.int64), device=DEV)
    ref = torch.tensor(np.asarray(ref_mesh.vertices, dtype=np.float32), device=DEV)
    assert cur.shape[0] == ref.shape[0], "reference topology must match the output"
    owner = torch.arange(cur.shape[0], device=DEV)
    _, _, crossed = segmentMeshIntersections(ref, cur, cur, faces, owner)
    return result, int(crossed.sum().item())


def test_v4_integration():
    print("  === trajectory guard OFF ===")
    off, off_cross = _run_shell(enable_trajectory_guard=False)
    print("    chamfer_l1:", round(off["fitted"]["chamfer_l1"], 6))
    print("    independent trajectory crossings:", off_cross)

    print("  === trajectory guard ON ===")
    on, on_cross = _run_shell(enable_trajectory_guard=True)
    print("    chamfer_l1:", round(on["fitted"]["chamfer_l1"], 6))
    print("    kept:", on["kept"])
    print("    reported trajectory crossings:", on["trajectory_crossing_vertices"])
    print("    independent trajectory crossings:", on_cross)

    assert off_cross > 20, (
        f"expected the unguarded shell collapse to cross trajectories; got {off_cross}"
    )
    assert on_cross == 0, (
        f"trajectory guard failed: {on_cross} independent crossings remain"
    )
    assert int(on["trajectory_crossing_vertices"]) == 0, (
        "fitter-reported trajectory crossings must be zero"
    )
    assert on["fitted"]["chamfer_l1"] < on["baseline"]["chamfer_l1"], (
        "guarded fit must still improve on the rigid baseline (not just revert)"
    )
    print("  test_v4_integration PASSED")


def main():
    print("[test] segmentMeshIntersections")
    test_segment_mesh_intersections()
    print("[test] local face scope")
    test_local_face_scope()
    print("[test] largestSafeStep hits + pullback")
    test_largest_safe_step_hits_and_pullback()
    print("[test] parametric atom t value + endpoint window")
    test_param_atom_t_value()
    print("[test] earliest-hit reduction (two sheets)")
    test_earliest_hit_reduction()
    print("[test] earliestSafeStep minimum pull-back")
    test_earliest_safe_step_min_pullback()
    print("[test] triton param kernel matches torch")
    test_triton_param_matches_torch()
    print("[test] v4 integration (double-layer collapse)")
    test_v4_integration()
    print("\nALL V4 TRAJECTORY GUARD TESTS PASSED")


if __name__ == "__main__":
    main()
