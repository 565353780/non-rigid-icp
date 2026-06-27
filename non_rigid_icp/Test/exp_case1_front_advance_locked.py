"""Lock-and-refine fitting: freeze fitted vertices, only optimize new ones.

The user's monotonic-error idea: after each penetration-relaxed projection step,
the vertices that were optimized are LOCKED and never moved again; the above-mean
subdivision then inserts new midpoint vertices (between locked, on-target
vertices) which are the ONLY vertices the next step moves. Because a new midpoint
starts between two on-target vertices and is then projected onto the target, its
distance can only shrink and the locked surface is untouched -- so the global fit
error is non-increasing across steps, by construction, while still adding zero new
self-intersections (locked verts are frozen during the penetration relaxation).

Saves, under output/case1_front_advance_locked/:
  - front_advance_locked_log.json   (initial + per-step metrics + refine log)
  - step_00.ply .. step_NN.ply      (full deformed meshes per step)
  - debug/bbox_*/initial_source_crop.ply
  - debug/bbox_*/step_00_crop.ply .. (per-step source crop in each bbox)
  - debug/bbox_*/target_crop.ply

Run (flux env, GPU 2):
  CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.exp_case1_front_advance_locked
"""

import os

from non_rigid_icp.Demo.watertight_fitter import data_dict
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


ROI_BBOXES = (
    {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
    {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
)


def main():
    n_steps = int(os.environ.get("N_STEPS", "4"))
    backoff = float(os.environ.get("RELAX_BACKOFF", "0.8"))
    relax_iters = int(os.environ.get("RELAX_ITERS", "60"))
    max_refine_faces = int(os.environ.get("MAX_REFINE_FACES", "1500000"))
    refine_mean_mult = float(os.environ.get("REFINE_MEAN_MULT", "1.0"))
    save_folder = "./output/case1_front_advance_locked/"

    src_path, tgt_path = data_dict["watertight_case1"]
    fitter = WatertightFitter(
        device="cuda",
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        trajectory_seg_chunk=200000,
        max_subdivisions=n_steps - 1,
        refine_above_mean=True,
        refine_mean_mult=refine_mean_mult,
        max_refine_faces=max_refine_faces,
        eval_bboxes=ROI_BBOXES,
        prefit_crop_bboxes=ROI_BBOXES,
        prefit_crop_mode="centroid",
        save_result_folder_path=save_folder,
    )
    fitter.loadMeshes(Mesh(src_path), Mesh(tgt_path))
    out = fitter.fitFrontAdvancingLockedRefineSteps(
        n_steps=n_steps,
        backoff=backoff,
        relax_iters=relax_iters,
        refine_mean_mult=refine_mean_mult,
        save_folder=save_folder,
        crop_eval=True,
        compute_chamfer=False,
    )

    print("\n==== initial source ====")
    init = out["initial"]
    for k in (
        "n_vertices", "n_faces",
        "bbox_0_crop_f1", "bbox_0_crop_fit_error_l1",
        "bbox_1_crop_f1", "bbox_1_crop_fit_error_l1",
    ):
        print(f"  {k} = {init.get(k)}")

    print("\n==== per-step (locked refine) ====")
    for rec in out["steps"]:
        print(
            f"  step {rec['step']} (lvl {rec['level']}, V={rec['n_vertices']}, "
            f"locked={rec['n_locked_after']}, moved={rec['n_moved']}): "
            f"unlocked_resid {rec['fit_residual_before_tau']:.3f}->"
            f"{rec['fit_residual_after_tau']:.3f}tau | "
            f"pen {rec['pen_pairs_start']}->{rec['pen_pairs_end']} | "
            f"tri_si_new={rec['tri_intersecting_pairs_new']} | "
            f"bbox_0_f1={rec.get('bbox_0_crop_f1')} "
            f"bbox_1_f1={rec.get('bbox_1_crop_f1')} | "
            f"refined={rec.get('refined')} -> "
            f"F={rec.get('faces_after_refine', rec['n_vertices'])}"
        )
    print(f"\n  total seconds = {out['seconds']}")


if __name__ == "__main__":
    main()
