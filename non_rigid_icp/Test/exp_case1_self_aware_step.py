"""Self-intersection-free single-step projection on case1 (two ROI bboxes).

The user's elegant idea: move every source vertex toward its EXACT target
closest point, but cap each vertex's advance at the earliest point where its
own segment v->cp would hit the (static) source mesh, minus a tiny clearance.
By construction no vertex tunnels a face and no two faces are driven to coincide,
so the single step is self-intersection-free.

This script validates that on case1 inside the two ROI bboxes (prefit-cropped for
a fast, faithful turnaround): it reports the bbox Chamfer-L1 + F1@tau and -- the
key check -- the number of NEW triangle-triangle self-intersections, which the
previous unconstrained single step produced by the millions and which this step
should drive to ~0.

Run (flux env, GPU 2):
  CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.exp_case1_self_aware_step

SAFE=<float> overrides safe_dist_percent (default 0.001). FULL=1 runs on the
whole mesh instead of the ROI crop.
"""

import json
import os
import time

from non_rigid_icp.Demo.watertight_fitter import data_dict
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


ROI_BBOXES = (
    {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
    {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
)


def main():
    safe = float(os.environ.get("SAFE", "0.001"))
    use_crop = os.environ.get("FULL", "0") != "1"

    src_path, tgt_path = data_dict["watertight_case1"]
    source_mesh = Mesh(src_path)
    target_mesh = Mesh(tgt_path)

    fitter = WatertightFitter(
        device="cuda",
        normal_gate=False,
        train_target_samples=2000000,
        eval_samples=500000,
        # isolate the new mechanism: every legacy guard OFF
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        trajectory_seg_chunk=200000,
        max_subdivisions=0,
        eval_bboxes=ROI_BBOXES,
        crop_eval_samples=300000,
        prefit_crop_bboxes=(ROI_BBOXES if use_crop else None),
        prefit_crop_mode="centroid",
        save_result_folder_path=f"./output/case1_self_aware/safe_{safe:g}/",
    )
    fitter.loadMeshes(source_mesh, target_mesh)

    print("=" * 78)
    print(f"[self-aware step] safe_dist_percent={safe}  "
          f"crop={'ROI' if use_crop else 'FULL MESH'}")
    print("=" * 78)
    t0 = time.time()
    rec = fitter.fitSelfAwareSingleStep(
        safe_dist_percent=safe,
        crop_eval=True,
        compute_chamfer=False,
    )
    elapsed = time.time() - t0

    print("\n" + "#" * 78)
    print("# RESULT (single self-aware projection step, inside the 2 ROI bboxes)")
    print("#" * 78)
    for box in ROI_BBOXES:
        name = box["name"]
        cd = rec.get(f"{name}_crop_chamfer_l1")
        f1 = rec.get(f"{name}_crop_f1")
        print(f"#   {name}: chamfer_l1={cd}  f1={f1}")
    print(f"#   residual mean: {rec['fit_residual_before_tau']:.3f}tau -> "
          f"{rec['fit_residual_after_tau']:.3f}tau")
    print(f"#   crossing segments (capped): {rec['n_crossing_segments']} / "
          f"{rec['n_vertices']}  (mean_alpha={rec['mean_alpha']:.4f})")
    print(f"#   NEW triangle-triangle self-intersections: "
          f"{rec['tri_intersecting_pairs_new']} "
          f"(total {rec['tri_intersecting_pairs_total']}, "
          f"faces {rec['tri_intersecting_faces']})")
    print(f"#   vertex-trajectory crossings: "
          f"{rec['trajectory_crossing_vertices']}")
    print(f"#   elapsed: {elapsed:.1f}s")
    print("#" * 78)

    tri_new = rec["tri_intersecting_pairs_new"]
    if tri_new == 0:
        print("[RESULT] PASS: the self-aware step created ZERO new "
              "self-intersections inside the ROIs.")
    else:
        print(f"[RESULT] {tri_new} new self-intersections remain -- the static-"
              "mesh approximation leaves residue; try a larger safe_dist_percent "
              "or a second self-aware iteration.")


if __name__ == "__main__":
    main()
