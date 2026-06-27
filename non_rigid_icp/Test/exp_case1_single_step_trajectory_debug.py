"""Single-step trajectory self-intersection debug on case1.

The user observed that a single step_frac=1.0 projection lands the watertight
source on the target but creates MANY self-intersections. This script is the
regression baseline that

  1. reproduces that: a single clamped step with step_frac=1.0 and the pull-back
     DISABLED (resolve_iters=0) -- the trajectory measurement must report a large
     number of crossing vertices/pairs and a small min crossing parameter t;
  2. verifies the fix: the same single step with the closed-form min-t pull-back
     ENABLED drives the full-mesh trajectory crossings to zero (or reports the
     exact residual + its hit type if a degenerate/pre-existing touch remains).

Both passes save the de-normalized step mesh + a JSON of the before/after
trajectory diagnostics, so the offending region can be inspected.

Run (flux env, GPU 2):
  CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.exp_case1_single_step_trajectory_debug

By default it uses the two ROI bboxes as a pre-fit crop for a fast turnaround;
set FULL=1 to validate on the whole 14M-vertex mesh (slower, most faithful).
"""

import json
import os
import time

import torch

from non_rigid_icp.Demo.watertight_fitter import data_dict
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


ROI_BBOXES = (
    {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
    {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
)


def _make_fitter(resolve_iters: int, tag: str, use_crop: bool):
    return WatertightFitter(
        device="cuda",
        laplacian_weight=30.0,
        normal_gate=False,
        train_target_samples=2000000,
        eval_samples=500000,
        # isolate the trajectory guard -- it is the only self-intersection
        # mechanism under test here.
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        trajectory_seg_chunk=200000,
        trajectory_clearance_tau=0.05,
        max_subdivisions=0,
        eval_bboxes=ROI_BBOXES,
        crop_eval_samples=300000,
        prefit_crop_bboxes=(ROI_BBOXES if use_crop else None),
        prefit_crop_mode="centroid",
        save_result_folder_path=f"./output/case1_traj_debug/{tag}/",
    )


def run_pass(resolve_iters: int, tag: str, use_crop: bool):
    src_path, tgt_path = data_dict["watertight_case1"]
    source_mesh = Mesh(src_path)
    target_mesh = Mesh(tgt_path)

    fitter = _make_fitter(resolve_iters, tag, use_crop)
    fitter.loadMeshes(source_mesh, target_mesh)

    t0 = time.time()
    out = fitter.fitStepwiseClamped(
        n_steps=1,
        step_frac=1.0,
        gd_lr=2.0,
        lap_w=30.0,
        resolve_iters=resolve_iters,
        max_subdivisions=0,
        converge_abs_tau=0.0,
        compute_chamfer=False,
        crop_eval=True,
        full_chamfer_each_step=False,
        save_full_each_step=True,
        refine_every_step=False,
        trajectory_debug=True,
    )
    rec = out["steps"][-1]
    elapsed = time.time() - t0

    folder = fitter.save_result_folder_path
    os.makedirs(folder, exist_ok=True)
    diag = {
        "tag": tag,
        "resolve_iters": resolve_iters,
        "use_crop": use_crop,
        "n_vertices": rec["n_vertices"],
        "n_faces": rec["n_faces"],
        "fit_residual_mean_tau": rec["fit_residual_mean_tau"],
        "trajectory_crossing_vertices_before": rec.get(
            "trajectory_crossing_vertices_before"
        ),
        "trajectory_crossing_pairs_before": rec.get(
            "trajectory_crossing_pairs_before"
        ),
        "trajectory_min_t_before": rec.get("trajectory_min_t_before"),
        "trajectory_repaired_vertices": rec.get("trajectory_repaired_vertices"),
        "trajectory_crossing_vertices_after": rec.get(
            "trajectory_crossing_vertices_after"
        ),
        "trajectory_crossing_pairs_after": rec.get(
            "trajectory_crossing_pairs_after"
        ),
        "trajectory_min_t_after": rec.get("trajectory_min_t_after"),
        "tri_intersecting_pairs_new": rec.get("tri_intersecting_pairs_new"),
        "tri_intersecting_pairs_total": rec.get("tri_intersecting_pairs_total"),
        "tri_intersecting_faces": rec.get("tri_intersecting_faces"),
        "elapsed_s": round(elapsed, 1),
    }
    with open(os.path.join(folder, "trajectory_debug.json"), "w") as f:
        json.dump(diag, f, indent=2)
    return diag


def main():
    use_crop = os.environ.get("FULL", "0") != "1"
    print("=" * 78)
    print(f"[case1 traj debug] crop={'ROI' if use_crop else 'FULL MESH'}")
    print("=" * 78)

    print("\n--- PASS 1: pull-back DISABLED (reproduce the crossings) ---")
    no_guard = run_pass(resolve_iters=0, tag="no_pullback", use_crop=use_crop)
    print(json.dumps(no_guard, indent=2))

    print("\n--- PASS 2: pull-back ENABLED (verify the fix) ---")
    guard = run_pass(resolve_iters=8, tag="with_pullback", use_crop=use_crop)
    print(json.dumps(guard, indent=2))

    print("\n" + "#" * 78)
    # vertex-trajectory crossings (what the trajectory guard checks/repairs)
    traj_before = no_guard["trajectory_crossing_vertices_after"]
    traj_after = guard["trajectory_crossing_vertices_after"]
    # triangle-triangle self-intersections (the OTHER failure mode: adjacent
    # faces folding/overlapping with no single vertex tunnelling a foreign face)
    tri_no_guard = no_guard["tri_intersecting_pairs_new"]
    tri_guard = guard["tri_intersecting_pairs_new"]
    print(f"# VERTEX-TRAJECTORY crossings  no-pullback={traj_before}  "
          f"with-pullback={traj_after}")
    print(f"# TRIANGLE-TRIANGLE new self-isect  no-pullback={tri_no_guard}  "
          f"with-pullback={tri_guard}  "
          f"(faces involved: {guard['tri_intersecting_faces']})")
    print("#" * 78)

    # The trajectory DETECTOR is proven correct by the unit tests. On this case1
    # single step the dominant self-intersection mode is decided here:
    if (traj_before or 0) > 0:
        print("[RESULT] vertex-trajectory crossings ARE present; "
              f"pull-back reduced {traj_before} -> {traj_after}.")
    elif (tri_no_guard or 0) > 0:
        print(
            "[RESULT] the single step produced ~0 vertex-trajectory crossings "
            f"but {tri_no_guard} NEW triangle-triangle self-intersections. The "
            "self-intersections the user sees are the triangle-triangle / face-"
            "fold mode, which the rest->current trajectory test cannot catch -- "
            "the next guard must act on current-face overlaps, not just vertex "
            "tunnelling."
        )
    else:
        print("[RESULT] no self-intersections of either kind on this region.")


if __name__ == "__main__":
    main()
