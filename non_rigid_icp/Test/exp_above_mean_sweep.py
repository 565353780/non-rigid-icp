"""Sweep the ABOVE-MEAN centroid-distance subdivision threshold over the FIRST 4
steps, to find the `mean_mult` that maximizes the dense-sampled Chamfer/F1 inside
the two ROI bboxes.

Locked setting (per user): step_frac=1.0 (single-step closest-point projection
== near-optimal vertex placement); refine_every_step=True (after each step, split
every face whose centroid distance to the target exceeds
mean_mult * mean_g d(centroid(g),T), then the next step projects the new
midpoints). Goal: the WHOLE source surface onto the target.

Compared at the LAST of the 4 steps: per-bbox crop Chamfer-L1 (lower better) and
F1 @ tau=L/2048 (higher better), plus the face count (cost).
"""

import math
import os
import time

from non_rigid_icp.Demo.watertight_fitter import data_dict
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


ROI_BBOXES = (
    {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
    {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
)


def run_one(mean_mult: float, n_steps: int):
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
        max_subdivisions=n_steps,          # one refine round per step
        refine_above_mean=True,
        refine_mean_mult=mean_mult,
        refine_min_centroid_tau=0.0,
        refine_ratio=0.0,
        refine_sag_mult=0.0,
        max_refine_faces=2000000,
        min_component_faces=1,
        dilation_rings=0,
        eval_bboxes=ROI_BBOXES,
        crop_eval_samples=300000,
        prefit_crop_bboxes=ROI_BBOXES,
        prefit_crop_mode="centroid",
        save_result_folder_path=f"./output/exp_above_mean/mult_{mean_mult:.2f}/",
    )
    fitter.loadMeshes(source_mesh, target_mesh)

    out = fitter.fitStepwiseClamped(
        n_steps=n_steps,
        step_frac=1.0,
        gd_lr=2.0,
        lap_w=30.0,
        max_subdivisions=n_steps,
        converge_abs_tau=0.0,
        compute_chamfer=False,
        crop_eval=True,
        full_chamfer_each_step=False,
        save_full_each_step=False,
        refine_every_step=True,
    )
    return out


def main():
    mults_env = os.environ.get("SWEEP_MULTS")
    if mults_env:
        mults = [float(x) for x in mults_env.split(",")]
    else:
        mults = [0.5, 1.0, 1.5, 2.0]
    n_steps = int(os.environ.get("SWEEP_NSTEPS", "4"))

    summary = {}
    for m in mults:
        print("\n" + "=" * 72)
        print(f"[EXP] refine_mean_mult = {m}")
        print("=" * 72)
        t0 = time.time()
        out = run_one(m, n_steps)
        summary[m] = {"recs": out["steps"], "elapsed": time.time() - t0}

    print("\n\n" + "#" * 94)
    print(f"# ABOVE-MEAN SWEEP -- metrics at the LAST of {n_steps} steps "
          "(ROI crop, step_frac=1.0)")
    print("#" * 94)
    hdr = (
        f"{'mult':>5} | {'n_faces':>9} | "
        f"{'b0_CD':>9} | {'b0_F1':>7} | {'b1_CD':>9} | {'b1_F1':>7} | "
        f"{'meanF1':>7} | {'sec':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    best = None
    for m in mults:
        last = summary[m]["recs"][-1]
        b0cd = last.get("bbox_0_crop_chamfer_l1")
        b0f1 = last.get("bbox_0_crop_f1")
        b1cd = last.get("bbox_1_crop_chamfer_l1")
        b1f1 = last.get("bbox_1_crop_f1")
        meanf1 = (
            (b0f1 + b1f1) / 2.0
            if (b0f1 is not None and b1f1 is not None) else float("nan")
        )
        print(
            f"{m:>5.2f} | {last['n_faces']:>9} | "
            f"{(b0cd if b0cd is not None else float('nan')):>9.2e} | "
            f"{(b0f1 if b0f1 is not None else float('nan')):>7.4f} | "
            f"{(b1cd if b1cd is not None else float('nan')):>9.2e} | "
            f"{(b1f1 if b1f1 is not None else float('nan')):>7.4f} | "
            f"{meanf1:>7.4f} | {summary[m]['elapsed']:>6.1f}"
        )
        if not math.isnan(meanf1) and (best is None or meanf1 > best[1]):
            best = (m, meanf1, last["n_faces"])
    if best is not None:
        print("-" * len(hdr))
        print(f"# BEST mean-F1: mult={best[0]:.2f}  meanF1={best[1]:.4f}  "
              f"n_faces={best[2]}")

    print("\n# per-step n_faces / mean-F1 / mean-centroid-dist(tau) trajectory")
    for m in mults:
        recs = summary[m]["recs"]
        traj = " ".join(
            f"[s{rr['step']}:F={rr['n_faces']},"
            f"f1={(((rr.get('bbox_0_crop_f1') or 0)+(rr.get('bbox_1_crop_f1') or 0))/2):.4f}]"
            for rr in recs
        )
        print(f"  mult {m:.2f}: {traj}")


if __name__ == "__main__":
    main()
