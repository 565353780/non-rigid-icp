"""40-step sag-driven adaptive-subdivision run with the fastest stable step
(step_frac=1.0). Validates the first-principles goal: the WHOLE source surface
on the target with the FEWEST faces, LOWEST error, and ZERO self-intersection.

Fast iteration: the full source+target are loaded, rigidly aligned, then cropped
in lock-step to the two ROI bboxes (WatertightFitter.prefit_crop_bboxes) so each
kept region is exactly what the full-mesh run would optimize. Set RUN_FULL=1 to
skip the crop and run the whole mesh.
"""

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
    run_full = os.environ.get("RUN_FULL", "0") == "1"
    n_steps = int(os.environ.get("NSTEPS", "40"))
    out_dir = os.environ.get(
        "OUT", "./output/case1_sag40" + ("_full" if run_full else "")
    )

    src_path, tgt_path = data_dict["watertight_case1"]
    source_mesh = Mesh(src_path)
    target_mesh = Mesh(tgt_path)

    fitter = WatertightFitter(
        device="cuda",
        laplacian_weight=30.0,
        normal_gate=False,
        train_target_samples=2000000,
        eval_samples=500000,
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        trajectory_seg_chunk=200000,
        # sag-driven adaptive subdivision
        max_subdivisions=12,
        plateau_window=1,
        plateau_rel_tol=5e-3,
        plateau_patience=1,
        refine_sag_mult=float(os.environ.get("SAG_MULT", "2.0")),
        refine_centroid_mult=float(os.environ.get("CENT_MULT", "2.0")),
        refine_sag_quantile=0.0,
        max_refine_faces=1500000,
        min_component_faces=1,
        dilation_rings=0,
        eval_bboxes=ROI_BBOXES,
        prefit_crop_bboxes=None if run_full else ROI_BBOXES,
        prefit_crop_mode="centroid",
        save_result_folder_path=out_dir,
    )
    fitter.loadMeshes(source_mesh, target_mesh)

    t0 = time.time()
    out = fitter.fitStepwiseClamped(
        n_steps=n_steps,
        step_frac=1.0,
        gd_lr=2.0,
        lap_w=30.0,
        max_subdivisions=12,
        converge_abs_tau=0.05,
        save_folder=out_dir,
        compute_chamfer=True,        # final FULL chamfer/F1
        crop_eval=True,
        full_chamfer_each_step=False,
        save_full_each_step=False,
    )
    print("\n[EXP] done in", round(time.time() - t0, 1), "s")
    print("[EXP] subdivision_rounds =", out["subdivision_rounds"])
    print("[EXP] final_chamfer_l1 =", out.get("final_chamfer_l1"),
          "final_f1 =", out.get("final_f1"))
    # face-count + self-intersection summary across steps
    recs = out["steps"]
    print("\n# step | level | n_faces | resid_tau | repaired | b0_f1 | b1_f1")
    for r in recs:
        print(
            f"  {r['step']:>2} | {r['level']:>2} | {r['n_faces']:>9} | "
            f"{r['fit_residual_mean_tau']:>8.3f} | "
            f"{r['trajectory_repaired_vertices']:>6} | "
            f"{r.get('bbox_0_crop_f1', float('nan')):.4f} | "
            f"{r.get('bbox_1_crop_f1', float('nan')):.4f}"
        )
    total_repaired = sum(r["trajectory_repaired_vertices"] for r in recs)
    print("\n[EXP] TOTAL self-intersection repairs across all steps:", total_repaired)
    print("[EXP] final n_faces:", recs[-1]["n_faces"])


if __name__ == "__main__":
    main()
