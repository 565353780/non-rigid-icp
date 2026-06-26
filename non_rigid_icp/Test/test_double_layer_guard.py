"""Double-layer collapse test: the real case1 failure mode in miniature.

Source: a concentric spherical SHELL (an inner sphere + an outer sphere a few
tau apart -- a genuine thin double layer). Target: a single sphere at the
shell's mid radius. Fitting pulls the outer shell inward and the inner shell
outward onto the SAME mid surface, so the inner shell must expand THROUGH the
outer shell -- a forced crossing. Without a guard this self-intersects heavily;
with the order / collision guard the layers stay nested and the result is
self-intersection free.

    python -m non_rigid_icp.Test.test_double_layer_guard
"""

import numpy as np
import torch
import open3d as o3d

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter
from non_rigid_icp.Method.self_intersection import findSelfIntersections


def _sphere(radius: float, resolution: int = 24, flip: bool = False):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    if flip:
        t = np.asarray(m.triangles)[:, ::-1].copy()
        m.triangles = o3d.utility.Vector3iVector(t)
    return m


def _shell(r_in: float, r_out: float, resolution: int = 24) -> o3d.geometry.TriangleMesh:
    outer = _sphere(r_out, resolution)
    inner = _sphere(r_in, resolution, flip=True)  # inward-facing inner layer
    ov = np.asarray(outer.vertices)
    ot = np.asarray(outer.triangles)
    iv = np.asarray(inner.vertices)
    it = np.asarray(inner.triangles) + ov.shape[0]
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.concatenate([ov, iv], axis=0))
    m.triangles = o3d.utility.Vector3iVector(np.concatenate([ot, it], axis=0))
    m.compute_vertex_normals()
    return m


def _run(enable_guard: bool) -> dict:
    # shell layers ~13 tau apart (within the 16-tau data mask), collapsing onto
    # the mid sphere -> the inner must pass through the outer.
    source = Mesh.from_o3d(_shell(r_in=0.447, r_out=0.453, resolution=30))
    target = Mesh.from_o3d(_sphere(0.450, resolution=40))
    fitter = WatertightFitter(
        device="cuda",
        outer_iter=120,
        inner_iter=8,
        # the sheet barrier is anti-tunnel only while a step stays below the
        # margin; lr=3e-3 (~7 tau/step) would jump straight through it. Keep the
        # step below the ~0.5 tau margin, matching the case1 configuration.
        lr=2e-4,
        train_source_samples=60000,
        train_target_samples=150000,
        eval_samples=150000,
        laplacian_weight=1.0,
        # force BOTH layers onto the target (disable the normal gate) so this
        # remains a hard stress test of the order/collision BARRIER under a
        # forced crossing -- normal-gated collapse-avoidance is validated on the
        # real case, not here.
        normal_gate=False,
        enable_self_collision_guard=enable_guard,
        enable_sheet_guard=enable_guard,
        enable_inversion_guard=enable_guard,
        collision_check_every=4,
        collision_refresh_every=4,
        sheet_gap_tau=20.0,
        sheet_max_thickness_tau=20.0,  # the shell wall is ~13.6 tau thick
        sheet_min_margin_tau=0.5,
        sheet_weight=200.0,
        collision_margin_tau=1.0,
        collision_broad_tau=8.0,
        max_subdivisions=0,
        save_result_folder_path=None,
        seed=0,
    )
    fitter.loadMeshes(source, target)
    result = fitter.fitAndEvaluate()
    # independently scan the deformed output (the fitter only runs its own scan
    # when the guard is enabled, so we must scan here for an apples-to-apples
    # comparison regardless of the guard flag).
    V = torch.tensor(
        np.asarray(fitter.source_mesh.vertices, dtype=np.float32), device="cuda"
    )
    F = torch.tensor(
        np.asarray(fitter.source_mesh.triangles, dtype=np.int64), device="cuda"
    )
    inter = findSelfIntersections(V, F, inflate=0.0, exclude_ring=1)
    return result, int(inter.shape[0])


def main():
    print("=== WITHOUT guard ===")
    no_guard, no_guard_si = _run(enable_guard=False)
    print("  chamfer_l1:", round(no_guard["fitted"]["chamfer_l1"], 6))
    print("  independent self-intersections:", no_guard_si)

    print("=== WITH guard ===")
    guard, guard_si = _run(enable_guard=True)
    print("  chamfer_l1:", round(guard["fitted"]["chamfer_l1"], 6))
    print("  kept (deformation accepted):", guard["kept"])
    print("  reported new self-intersections:", guard["final_new_self_intersections"])
    print("  independent self-intersections:", guard_si)

    assert no_guard_si > 100, (
        "expected the unguarded shell collapse to self-intersect heavily; "
        f"got {no_guard_si}"
    )
    assert guard_si == 0, (
        f"guard failed to prevent shell crossing: {guard_si} new intersections"
    )
    # the guard must ENABLE a good fit, not merely revert to the rigid init: the
    # double layer should straddle the target (each ~margin/2 from it), so the
    # guarded chamfer must be far below the rigid baseline and within a small
    # factor of the (self-intersecting) unguarded fit.
    assert guard["kept"], (
        "guarded deformation was rejected (reverted to rigid init); the barrier "
        "is over-constraining or tunneling instead of enabling a clean fit"
    )
    assert guard["fitted"]["chamfer_l1"] < 5.0 * no_guard["fitted"]["chamfer_l1"], (
        f"guarded fit too poor: {guard['fitted']['chamfer_l1']:.6f} vs unguarded "
        f"{no_guard['fitted']['chamfer_l1']:.6f}"
    )
    print("\nDOUBLE-LAYER GUARD TEST PASSED")


if __name__ == "__main__":
    main()
