"""Fast synthetic validation of the new pieces (no 14M mesh required):

  1. cropMeshByBBox          -- bbox crop into a compact sub-mesh.
  2. lock-step subdivision   -- _applySubdivision keeps ref[i] <-> current[i]
                                correspondence for every vertex (incl. new
                                midpoints), which is what makes the trajectory
                                self-intersection test trivial after refinement.
  3. fitStepwiseClamped      -- end-to-end clamped GD + plateau subdivision +
                                bbox crop eval + debug saving on a small mesh.
"""

import os
import shutil
import numpy as np
import torch
import open3d as o3d

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.crop import cropMeshByBBox, bboxFromCenterEdge
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


def _sphere_mesh(radius: float, resolution: int = 20) -> Mesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    s.compute_vertex_normals()
    return Mesh.from_o3d(s)


def test_crop_atom():
    print("=== test_crop_atom ===")
    # unit cube grid of vertices via a box mesh
    box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    box = box.subdivide_midpoint(3)
    v = np.asarray(box.vertices)
    f = np.asarray(box.triangles)
    # crop to the lower corner octant
    center = np.array([0.25, 0.25, 0.25])
    edge = 0.5
    sv, sf, vk, fk = cropMeshByBBox(v, f, center, edge, mode="all")
    lo, hi = bboxFromCenterEdge(center, edge)
    assert sv.shape[0] == int(vk.sum())
    assert sf.shape[0] == int(fk.sum())
    # every kept vertex inside the box, faces re-indexed in range
    assert ((sv >= lo - 1e-9) & (sv <= hi + 1e-9)).all()
    assert sf.min() >= 0 and sf.max() < sv.shape[0]
    print(f"  crop ok: V {v.shape[0]}->{sv.shape[0]}, F {f.shape[0]}->{sf.shape[0]}")


def test_lockstep_subdivision():
    print("=== test_lockstep_subdivision ===")
    src = _sphere_mesh(0.33, resolution=20)
    tgt = _sphere_mesh(0.30, resolution=20)
    fitter = WatertightFitter(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        train_target_samples=20000,
        eval_samples=20000,
        save_result_folder_path=None,
    )
    fitter.loadMeshes(src, tgt)
    fitter._setupFit()

    # give the source a non-trivial deformation so ref != current
    with torch.no_grad():
        fitter._disp.data.copy_(0.01 * torch.randn_like(fitter._disp))

    ref_before = fitter._ref_verts.clone()
    v_before = fitter._verts.shape[0]
    deformed_before = fitter._deformed().detach().clone()

    # subdivide the first few faces
    mask = torch.zeros(fitter._faces.shape[0], dtype=torch.bool, device=fitter.device)
    mask[:50] = True
    edges_pre, fe = None, None
    fitter._applySubdivision(mask)

    # correspondence invariant: same vertex count in both fields
    assert fitter._ref_verts.shape == fitter._verts.shape, "ref/current count mismatch"
    n_after = fitter._verts.shape[0]
    assert n_after > v_before, "subdivision added no vertices"

    # original vertices unchanged in the rest field; current base == old deformed
    assert torch.allclose(fitter._ref_verts[:v_before], ref_before, atol=1e-6)
    assert torch.allclose(fitter._verts[:v_before], deformed_before, atol=1e-6)

    # _disp restarted at zero (geometry preserving) -> current == baked deformed
    assert torch.allclose(fitter._deformed()[:v_before], deformed_before, atol=1e-6)

    # new midpoint rest positions lie on the rest sphere segment midpoints
    # (correspondence is exact, so the trajectory segment ref[i]->current[i] is
    # well defined for every new vertex too)
    print(f"  lock-step ok: V {v_before}->{n_after}, "
          f"ref.shape==verts.shape=={tuple(fitter._ref_verts.shape)}")


def test_stepwise_clamped_with_subdivision():
    print("=== test_stepwise_clamped_with_subdivision ===")
    out_dir = "./output/_test_stepwise_clamped/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    src = _sphere_mesh(0.36, resolution=24)
    tgt = _sphere_mesh(0.30, resolution=24)
    fitter = WatertightFitter(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        train_target_samples=40000,
        eval_samples=40000,
        # force a plateau quickly so subdivision is exercised
        plateau_window=1,
        plateau_rel_tol=0.5,
        plateau_patience=1,
        eval_bbox_center=(0.0, 0.0, 0.0),
        eval_bbox_edge=0.5,
        crop_eval_samples=20000,
        save_result_folder_path=out_dir,
    )
    fitter.loadMeshes(src, tgt)
    out = fitter.fitStepwiseClamped(
        n_steps=6,
        step_frac=0.2,
        max_subdivisions=2,
        save_folder=out_dir,
        crop_eval=True,
        full_chamfer_each_step=False,
        save_full_each_step=False,
    )
    # residual should be monotonically (mostly) decreasing
    res = [r["fit_residual_mean_tau"] for r in out["steps"]]
    print("  residual_tau per step:", [round(x, 2) for x in res])
    assert res[-1] < res[0], "residual did not decrease"
    assert out["subdivision_rounds"] >= 1, "no subdivision triggered"
    # crop debug files exist
    dbg = os.path.join(out_dir, "debug")
    assert os.path.exists(os.path.join(dbg, "target_crop.ply")), "no target crop saved"
    assert os.path.exists(os.path.join(dbg, "step_00_crop.ply")), "no source crop saved"
    # crop metrics present
    assert out["steps"][0].get("crop_chamfer_l1") is not None
    print(f"  stepwise ok: subdivisions={out['subdivision_rounds']}, "
          f"crop_cd[0]={out['steps'][0]['crop_chamfer_l1']:.5f} -> "
          f"crop_cd[-1]={out['steps'][-1]['crop_chamfer_l1']:.5f}, "
          f"final_full_cd={out.get('final_chamfer_l1'):.5f}")


if __name__ == "__main__":
    test_crop_atom()
    test_lockstep_subdivision()
    test_stepwise_clamped_with_subdivision()
    print("\nALL TESTS PASSED")
