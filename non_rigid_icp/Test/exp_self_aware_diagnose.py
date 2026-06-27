"""Root-cause diagnosis of the self-aware single step still self-intersecting.

The single self-aware projection step reports n_crossing_segments=0 / min_t=1.0
yet leaves ~10M triangle-triangle self-intersections. Either the segment-mesh
detector is blind to the real motion (a bug), or the colliding mode is genuinely
not vertex-segment (faces folding). This script settles it on the ROI crop by
INSTRUMENTING the exact projection math and then dissecting the survivors:

  A. Sensitivity of the segment-mesh detector to t_lo: count crossing segments
     with t_lo in {0, 1e-6, 1e-4}. If 0 -> many, 1e-4 -> 0, the t_lo filter is
     the bug (it discards near-rest crossings the cap should have honored).

  B. For the post-step intersecting face pairs, the penetration depth: classify
     each as TRUE interpenetration vs NEAR-COPLANAR/touch (depth <= eps*tau).
     A wall of "touch" pairs means the t-0.001 PARAMETRIC clearance collapsed the
     two layers to ~0 geometric gap, not a tunnelling failure.

  C. For a sample of intersecting pairs, whether ANY of the 6 vertices'
     cur->cp segments actually pierce the OTHER triangle at its PRE-step (cur)
     position. If essentially none do, the collision is face-fold / two-sheets-
     converging, which a vertex-segment test cannot catch by construction.

Run (flux env, GPU 2):
  CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.exp_self_aware_diagnose
"""

import numpy as np
import torch

from non_rigid_icp.Demo.watertight_fitter import data_dict
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter
from non_rigid_icp.Method.implicit_field import ImplicitField
from non_rigid_icp.Method.trajectory_guard import (
    segmentMeshIntersectionParams,
    earliestSegmentMeshHits,
)
from non_rigid_icp.Method.self_intersection import findSelfIntersections
from non_rigid_icp.Method.geometry import segmentTriangleIntersect


ROI_BBOXES = (
    {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
    {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
)
DEV = "cuda"


def _earliest_t(cur, cp, faces, owner, cell, chunk, t_lo):
    v = cur.shape[0]
    t_min = torch.full((v,), float("inf"), device=cur.device)
    for s in range(0, v, chunk):
        sub = owner[s:s + chunk]
        hs, hf, ht = segmentMeshIntersectionParams(
            cur[sub], cp[sub], cur, faces, owner_vid=sub,
            inflate=0.0, cell_size=cell, face_ids=None, t_lo=t_lo, t_hi=1.0,
        )
        tsub, _ = earliestSegmentMeshHits(hs, hf, ht, sub.numel())
        t_min[s:s + chunk] = tsub
    return t_min


def main():
    src_path, tgt_path = data_dict["watertight_case1"]
    fitter = WatertightFitter(
        device=DEV,
        normal_gate=False,
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        trajectory_seg_chunk=200000,
        max_subdivisions=0,
        eval_bboxes=ROI_BBOXES,
        prefit_crop_bboxes=ROI_BBOXES,
        prefit_crop_mode="centroid",
        save_result_folder_path="./output/self_aware_diag/",
    )
    fitter.loadMeshes(Mesh(src_path), Mesh(tgt_path))
    fitter._setupFit()

    dev = fitter.device
    tau = fitter._tau_norm
    tV = np.asarray(fitter.target_mesh.vertices, dtype=np.float32)
    tF = np.asarray(fitter.target_mesh.triangles)
    field = ImplicitField(tV, tF, device=dev)

    faces = fitter._faces
    cur0 = fitter._deformed().detach().clone()  # pre-step source (normalized)
    cp, _, _ = field.closestPoints(cur0)
    d_i = (cp - cur0).norm(dim=1)
    v = cur0.shape[0]
    owner = torch.arange(v, device=dev)
    cell = fitter._trajCellSize(cur0)
    chunk = fitter.trajectory_seg_chunk

    print("=" * 78)
    print(f"V={v}  F={faces.shape[0]}  tau(norm)={tau:.6e}")
    print(f"mean d_i={float(d_i.mean()):.6e} ({float(d_i.mean())/tau:.3f} tau), "
          f"max d_i={float(d_i.max())/tau:.3f} tau")
    print("=" * 78)

    # ---- A. detector sensitivity to t_lo ----
    print("\n[A] segment-mesh crossing count vs t_lo "
          "(segments cur->cp against the STATIC pre-step mesh):")
    for t_lo in (0.0, 1e-8, 1e-6, 1e-4):
        t_min = _earliest_t(cur0, cp, faces, owner, cell, chunk, t_lo)
        nc = int(torch.isfinite(t_min).sum())
        finite = t_min[torch.isfinite(t_min)]
        mn = float(finite.min()) if finite.numel() else float("nan")
        print(f"    t_lo={t_lo:<7g}: crossing_segments={nc:>9d}  "
              f"min_t={mn:.3e}")

    # ---- A2. the DECISIVE test: segments ref->cp against the POST-MOVE mesh
    #          (every vertex already at cp). If many cross here, the detector is
    #          fine but the front-advancing round-1 test (against the STATIC
    #          rest mesh) is the wrong substrate -- it declares everyone clear. ----
    print("\n[A2] crossing count of ref->cp segments against the POST-MOVE "
          "(all-at-cp) mesh:")
    cell_cp = fitter._trajCellSize(cur0)  # cached; rebuild on cp below
    for t_lo in (0.0, 1e-4):
        v = cur0.shape[0]
        t_min = torch.full((v,), float("inf"), device=dev)
        for s in range(0, v, chunk):
            sub = owner[s:s + chunk]
            hs, hf, ht = segmentMeshIntersectionParams(
                cur0[sub], cp[sub], cp, faces, owner_vid=sub,
                inflate=0.0, cell_size=cell_cp, face_ids=None, t_lo=t_lo, t_hi=1.0,
            )
            tsub, _ = earliestSegmentMeshHits(hs, hf, ht, sub.numel())
            t_min[s:s + chunk] = tsub
        nc = int(torch.isfinite(t_min).sum())
        print(f"    t_lo={t_lo:<7g}: crossing_segments={nc:>9d}")

    # ---- perform the actual step (t_lo=1e-4, safe=0.001), then dissect ----
    t_min = _earliest_t(cur0, cp, faces, owner, cell, chunk, 1e-4)
    crossed = torch.isfinite(t_min)
    t_eff = torch.where(crossed, t_min, torch.ones_like(t_min))
    alpha = torch.clamp(t_eff - 0.001, min=0.0, max=1.0)
    new_pos = cur0 + alpha.unsqueeze(1) * (cp - cur0)

    # geometric clearance the parametric safe margin actually left, per vertex
    seg_len = (cp - cur0).norm(dim=1)
    clearance = 0.001 * seg_len  # = (t - alpha) * |seg|
    print("\n[clearance] geometric back-off left by safe=0.001 (parametric):")
    print(f"    mean={float(clearance.mean()):.3e} "
          f"({float(clearance.mean())/tau:.5f} tau), "
          f"max={float(clearance.max())/tau:.5f} tau")

    # ---- B. classify post-step intersections by penetration depth ----
    print("\n[B] post-step triangle-triangle self-intersections "
          "(authoritative findSelfIntersections):")
    inter = findSelfIntersections(
        new_pos, faces, inflate=0.0, exclude_ring=1,
        face_adjacency=fitter._face_adj,
    )
    print(f"    intersecting pairs = {inter.shape[0]}")
    if inter.shape[0] == 0:
        print("    (none -- nothing to dissect)")
        return

    # penetration proxy: for each intersecting pair, the min distance between
    # their planes' vertices is ~0 if merely touching/coplanar. Use the
    # centroid-to-centroid distance and the per-pair vertex spread vs tau.
    sample = inter[torch.randperm(inter.shape[0], device=dev)[:200000]]
    tri_a = new_pos[faces[sample[:, 0]]]  # (S,3,3)
    tri_b = new_pos[faces[sample[:, 1]]]
    ca = tri_a.mean(dim=1)
    cb = tri_b.mean(dim=1)
    cdist = (ca - cb).norm(dim=1)
    print(f"    [sample {sample.shape[0]}] centroid-centroid dist: "
          f"mean={float(cdist.mean())/tau:.4f} tau, "
          f"median={float(cdist.median())/tau:.4f} tau, "
          f"<0.5tau frac={float((cdist<0.5*tau).float().mean()):.3f}")

    # ---- C. do the colliding pairs' vertices' cur->cp segments pierce the
    #         OTHER triangle at its PRE-step position? ----
    print("\n[C] for sampled intersecting pairs, does ANY of the 6 vertices' "
          "cur->cp segment pierce the OTHER triangle at its PRE-STEP (cur) pose?")
    samp = sample[:50000]
    fa = faces[samp[:, 0]]  # (S,3)
    fb = faces[samp[:, 1]]
    # triangle B at pre-step pose
    b0 = cur0[fb[:, 0]]
    b1 = cur0[fb[:, 1]]
    b2 = cur0[fb[:, 2]]
    a0 = cur0[fa[:, 0]]
    a1 = cur0[fa[:, 0]]  # placeholder, overwritten below
    pierce_any = torch.zeros(samp.shape[0], dtype=torch.bool, device=dev)
    # each of A's 3 vertices vs triangle B
    for k in range(3):
        vid = fa[:, k]
        hit = segmentTriangleIntersect(cur0[vid], cp[vid], b0, b1, b2)
        pierce_any |= hit
    # each of B's 3 vertices vs triangle A (pre-step)
    ta0 = cur0[fa[:, 0]]
    ta1 = cur0[fa[:, 1]]
    ta2 = cur0[fa[:, 2]]
    for k in range(3):
        vid = fb[:, k]
        hit = segmentTriangleIntersect(cur0[vid], cp[vid], ta0, ta1, ta2)
        pierce_any |= hit
    frac = float(pierce_any.float().mean())
    print(f"    sampled pairs = {samp.shape[0]}")
    print(f"    fraction whose vertex cur->cp segment pierces the partner's "
          f"PRE-step triangle = {frac:.4f}")

    # ---- C2. classify intersecting pairs at the POST-MOVE pose: vertex-in-face
    #          (any of 6 vertices strictly inside the other triangle's slab) vs
    #          pure EDGE-EDGE (no vertex inside, only edges cross). A vertex-
    #          trajectory test can only ever catch the former. ----
    from non_rigid_icp.Method.geometry import segmentTriangleIntersect as _sti
    pa3 = new_pos[fa]  # (S,3,3) post-move triangle A
    pb3 = new_pos[fb]
    a0p, a1p, a2p = pa3[:, 0], pa3[:, 1], pa3[:, 2]
    b0p, b1p, b2p = pb3[:, 0], pb3[:, 1], pb3[:, 2]
    # a tri's edges crossing the other tri (edge-face), at the post-move pose
    edge_cross = torch.zeros(samp.shape[0], dtype=torch.bool, device=dev)
    for (e0, e1) in ((a0p, a1p), (a1p, a2p), (a2p, a0p)):
        edge_cross |= _sti(e0, e1, b0p, b1p, b2p)
    for (e0, e1) in ((b0p, b1p), (b1p, b2p), (b2p, a0p if False else b0p)):
        edge_cross |= _sti(e0, e1, a0p, a1p, a2p)
    print(f"    [post-move] pairs caught by some EDGE crossing the other face = "
          f"{float(edge_cross.float().mean()):.4f}")

    # ---- D. is the collapse driven by unsigned closest-point matching? ----
    # If two opposite sheets both project to the SAME target face, the two
    # layers converge onto one surface and overlap. Probe (a) signed side of
    # the pre-step source vertices wrt target, (b) whether intersecting pairs
    # share the same target primitive their cp landed on, (c) whether the two
    # faces of a pair point in OPPOSITE directions (back-to-back sheet).
    print("\n[D] is the overlap a double-sheet collapse (unsigned cp)?")
    _, prim_ids, _ = field.closestPoints(cur0)
    pa = prim_ids[fa]  # (S,3) target prim each A-vertex projected to
    pb = prim_ids[fb]
    # do the two faces' vertices project onto a shared target primitive set?
    same_prim = torch.zeros(samp.shape[0], dtype=torch.bool, device=dev)
    for ka in range(3):
        for kb in range(3):
            same_prim |= (pa[:, ka] == pb[:, kb])
    print(f"    pairs whose A & B vertices share >=1 target primitive cp = "
          f"{float(same_prim.float().mean()):.4f}")
    # back-to-back? compare the two source-face normals at pre-step pose
    na = torch.cross(ta1 - ta0, ta2 - ta0, dim=1)
    nb = torch.cross(cur0[fb[:, 1]] - cur0[fb[:, 0]],
                     cur0[fb[:, 2]] - cur0[fb[:, 0]], dim=1)
    na = na / (na.norm(dim=1, keepdim=True) + 1e-20)
    nb = nb / (nb.norm(dim=1, keepdim=True) + 1e-20)
    cosang = (na * nb).sum(dim=1)
    print(f"    face-pair normal cos: mean={float(cosang.mean()):.3f}, "
          f"back-to-back (cos<-0.5) frac={float((cosang<-0.5).float().mean()):.3f}, "
          f"aligned (cos>0.5) frac={float((cosang>0.5).float().mean()):.3f}")
    # signed side of the pre-step vertices (sample): mix of +/- => double layer
    sd = field.signedDistance(cur0[owner[:300000]])
    print(f"    pre-step signed dist sample: inside(<0) frac="
          f"{float((sd<0).float().mean()):.3f}, "
          f"|phi| mean={float(sd.abs().mean())/tau:.3f} tau")

    # ---- E. DECISIVE for the signed-separation fix: for each intersecting
    #         pair, are its two faces on OPPOSITE signed sides of the target?
    #         (then a signed offset separates them). Compute the signed side of
    #         every source vertex once, then per-pair compare the two faces'
    #         mean vertex sign. ----
    print("\n[E] signed-side separability of the colliding pairs:")
    sd_all = field.signedDistance(cur0)  # (V,) phi at pre-step
    sa = sd_all[fa].mean(dim=1)  # mean signed dist of face A vertices
    sb = sd_all[fb].mean(dim=1)
    opp = (sa.sign() != sb.sign())
    print(f"    pairs whose two faces are on OPPOSITE signed sides = "
          f"{float(opp.float().mean()):.4f}")
    print(f"    |phi(A)| mean={float(sa.abs().mean())/tau:.3f} tau, "
          f"|phi(B)| mean={float(sb.abs().mean())/tau:.3f} tau")
    same_close = (~opp) & (sa.abs() < 0.5 * tau)
    print(f"    SAME-side & both within 0.5tau of surface frac="
          f"{float(same_close.float().mean()):.4f} "
          f"(these a signed offset CANNOT separate)")

    # ---- F. DECISIVE: how many of the Moller 'intersections' are TRUE
    #         penetrations (some edge strictly crosses the other interior) vs
    #         mere near-coplanar CONTACT that the Moller coplanar/touch fallback
    #         miscounts? Use the exact edge-based predicate. ----
    from non_rigid_icp.Method.geometry import (
        triangleTrianglePenetrate, triangleTriangleIntersects,
    )
    print("\n[F] Moller-count vs EXACT edge-penetration count on ALL pairs:")
    n_moller = 0
    n_pen = 0
    CH = 4_000_000
    for s in range(0, inter.shape[0], CH):
        sub = inter[s:s + CH]
        t1 = new_pos[faces[sub[:, 0]]]
        t2 = new_pos[faces[sub[:, 1]]]
        n_moller += int(triangleTriangleIntersects(t1, t2).sum())
        n_pen += int(triangleTrianglePenetrate(t1, t2, eps=0.0).sum())
    print(f"    Moller intersecting pairs   = {n_moller}")
    print(f"    EXACT penetrating pairs     = {n_pen}")
    print(f"    -> {n_moller - n_pen} pairs are near-coplanar CONTACT, "
          f"not true penetration")
    print("\n[INTERPRETATION]")
    if frac < 0.05:
        print("    -> The colliding faces are NOT created by any vertex tunnelling")
        print("       a static partner face. The vertex-segment test is blind to")
        print("       them BY CONSTRUCTION (two sheets converging / faces folding,")
        print("       a moving-vs-moving collision). A vertex-segment cap cannot")
        print("       prevent this; a face-face / edge-edge swept (CCD) constraint")
        print("       or a normal/side-aware correspondence is required.")
    else:
        print("    -> A meaningful fraction DOES pierce a static partner triangle,")
        print("       so the detector SHOULD have flagged them. That points to a")
        print("       detector bug (t_lo filtering, broad-phase miss, or the")
        print("       owner/incident exclusion dropping real partners).")


if __name__ == "__main__":
    main()
