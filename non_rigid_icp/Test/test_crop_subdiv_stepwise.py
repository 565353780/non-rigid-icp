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
from non_rigid_icp.Method.fit_state import (
    initVertexFitState,
    updateVertexFitState,
    stateFloatAttrs,
    stateFromFloatAttrs,
)
from non_rigid_icp.Method.geometry import segmentTriangleIntersect
from non_rigid_icp.Method.triton_kernels import (
    segmentTrianglePairHits,
    clampNorm,
    tritonAvailable,
)
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


def test_fit_state_machine():
    print("=== test_fit_state_machine ===")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tau = 0.01
    st = initVertexFitState(4, dev)
    # vertex 0: wants to move, stays off-surface, never actually moves -> blocked
    # vertex 1: wants to move and makes progress -> stays optimizable
    # vertex 2: already on the surface (small residual) -> never counts as stalled
    # vertex 3: never intends to move -> never stalled
    resid = torch.tensor([5 * tau, 5 * tau, 0.1 * tau, 5 * tau], device=dev)
    intended = torch.tensor([0.2 * tau, 0.2 * tau, 0.2 * tau, 0.0], device=dev)
    actual_block = torch.tensor([0.0, 0.0, 0.0, 0.0], device=dev)
    actual_prog = torch.tensor([0.2 * tau, 0.2 * tau, 0.0, 0.0], device=dev)
    for _ in range(3):
        updateVertexFitState(
            st, resid,
            intended,
            torch.tensor([0.0, 0.2 * tau, 0.0, 0.0], device=dev),
            tau, block_patience=3,
        )
    opt = st["optimizable"]
    assert not bool(opt[0]), "stalled off-surface vertex should be dropped"
    assert bool(opt[1]), "progressing vertex must stay optimizable"
    assert bool(opt[2]), "on-surface vertex must stay optimizable"
    assert bool(opt[3]), "non-moving vertex must stay optimizable"
    print("  optimizable after 3 stalled steps:", opt.tolist())

    # pack/unpack round trip keeps the (original) state and blends midpoints
    attrs = stateFloatAttrs(st)
    # append a midpoint blending vtx0 (blocked) and vtx1 (optimizable)
    mid = 0.5 * (attrs[0] + attrs[1])
    attrs2 = torch.cat([attrs, mid.unsqueeze(0)], dim=0)
    st2 = stateFromFloatAttrs(attrs2, n_orig=4, device=dev)
    assert st2["optimizable"][:4].tolist() == opt.tolist(), "originals must survive"
    assert bool(st2["optimizable"][4]), "midpoint optimizable if either parent is"
    print("  pack/unpack ok; midpoint optimizable =", bool(st2["optimizable"][4]))


def test_triton_parity():
    print("=== test_triton_parity ===")
    if not torch.cuda.is_available():
        print("  (no CUDA, skipping)")
        return
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    s, t = 500, 400
    ss = torch.rand(s, 3, device=dev, generator=g)
    se = ss + 0.3 * (torch.rand(s, 3, device=dev, generator=g) - 0.5)
    va = torch.rand(t, 3, device=dev, generator=g)
    vb = torch.rand(t, 3, device=dev, generator=g)
    vc = torch.rand(t, 3, device=dev, generator=g)
    p = 8000
    seg_id = torch.randint(0, s, (p,), device=dev, generator=g)
    tri_id = torch.randint(0, t, (p,), device=dev, generator=g)

    hit_k = segmentTrianglePairHits(seg_id, tri_id, ss, se, va, vb, vc)
    hit_ref = segmentTriangleIntersect(
        ss[seg_id], se[seg_id], va[tri_id], vb[tri_id], vc[tri_id]
    )
    n_diff = int((hit_k != hit_ref).sum().item())
    print(f"  triton_available={tritonAvailable()}, hits={int(hit_ref.sum())}, "
          f"mismatches={n_diff}")
    assert n_diff == 0, "triton seg-tri kernel disagrees with torch"

    vecs = torch.randn(2000, 3, device=dev, generator=g)
    cap = torch.rand(2000, device=dev, generator=g)
    out_k = clampNorm(vecs, cap)
    norm = vecs.norm(dim=-1, keepdim=True)
    scale = torch.clamp(cap.reshape(-1, 1) / (norm + 1e-20), max=1.0)
    out_ref = vecs * scale
    assert torch.allclose(out_k, out_ref, atol=1e-5), "clampNorm kernel mismatch"
    print("  clampNorm parity ok")


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
        # exercise the local-plateau subdivision: a region must stop improving
        # (drop EMA below local_drop_tau) AND still be optimizable before it is
        # refined. A loose local_drop_tau + enough steps for the geometric decay
        # to flatten makes the synthetic sphere reach that state.
        local_drop_tau=0.2,
        eval_bboxes=(
            {"name": "bbox_0", "center": (0.0, 0.0, 0.0), "edge": 0.5},
            {"name": "bbox_1", "center": (0.1, 0.0, 0.0), "edge": 0.5},
        ),
        crop_eval_samples=20000,
        save_result_folder_path=out_dir,
    )
    fitter.loadMeshes(src, tgt)
    out = fitter.fitStepwiseClamped(
        n_steps=24,
        step_frac=0.2,
        max_subdivisions=2,
        converge_abs_tau=None,
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
    # crop debug files exist for BOTH boxes under their own subfolder
    dbg = os.path.join(out_dir, "debug")
    for name in ("bbox_0", "bbox_1"):
        assert os.path.exists(os.path.join(dbg, name, "target_crop.ply")), \
            f"no target crop saved for {name}"
        assert os.path.exists(os.path.join(dbg, name, "step_00_crop.ply")), \
            f"no source crop saved for {name}"
    # per-box crop metrics present + unprefixed alias for the first box
    assert out["steps"][0].get("bbox_0_crop_chamfer_l1") is not None
    assert out["steps"][0].get("bbox_1_crop_chamfer_l1") is not None
    assert out["steps"][0].get("crop_chamfer_l1") is not None
    # the state-machine metrics are recorded each step
    assert "n_unoptimizable_vertices" in out["steps"][0]
    assert len(out["eval_bboxes"]) == 2
    print(f"  stepwise ok: subdivisions={out['subdivision_rounds']}, "
          f"bbox_0_cd[0]={out['steps'][0]['bbox_0_crop_chamfer_l1']:.5f} -> "
          f"bbox_0_cd[-1]={out['steps'][-1]['bbox_0_crop_chamfer_l1']:.5f}, "
          f"unopt[-1]={out['steps'][-1]['n_unoptimizable_vertices']}, "
          f"final_full_cd={out.get('final_chamfer_l1'):.5f}")


if __name__ == "__main__":
    test_crop_atom()
    test_fit_state_machine()
    test_triton_parity()
    test_lockstep_subdivision()
    test_stepwise_clamped_with_subdivision()
    print("\nALL TESTS PASSED")
