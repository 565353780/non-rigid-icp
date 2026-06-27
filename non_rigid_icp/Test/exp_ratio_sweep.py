"""Sweep the centroid/vertex distance RATIO subdivision threshold over the FIRST
4 steps to find the value that maximizes the dense-sampled Chamfer/F1 inside the
two ROI bboxes.

Setting (locked, per user): step_frac=1.0 (single-step closest-point projection
== near-optimal vertex placement), refine_every_step=True (after each step,
split every face whose d(centroid,T)/mean d(verts,T) > refine_ratio AND
d(centroid,T) > centroid_mult*tau, then the next step projects the new
midpoints). Goal: get the WHOLE source surface onto the target.

We compare, at the LAST of the 4 steps, the per-bbox crop Chamfer-L1 (lower
better) and F1 @ tau=L/2048 (higher better), plus the face count (cost).
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


def run_one(ratio: float, n_steps: int):
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
        # RATIO-driven subdivision, every step (level-capped only).
        max_subdivisions=n_steps,          # allow one refine round per step
        refine_ratio=ratio,
        refine_centroid_mult=1.0,
        refine_ratio_denom_eps_tau=0.25,
        refine_sag_mult=0.0,               # disable the sag-DIFF path
        max_refine_faces=2000000,
        min_component_faces=1,
        dilation_rings=0,
        eval_bboxes=ROI_BBOXES,
        crop_eval_samples=300000,
        prefit_crop_bboxes=ROI_BBOXES,
        prefit_crop_mode="centroid",
        save_result_folder_path=f"./output/exp_ratio/ratio_{ratio:.2f}/",
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
    ratios_env = os.environ.get("SWEEP_RATIOS")
    if ratios_env:
        ratios = [float(x) for x in ratios_env.split(",")]
    else:
        ratios = [1.05, 1.1, 1.2, 1.4, 1.8]
    n_steps = int(os.environ.get("SWEEP_NSTEPS", "4"))

    summary = {}
    for r in ratios:
        print("\n" + "=" * 72)
        print(f"[EXP] refine_ratio = {r}")
        print("=" * 72)
        t0 = time.time()
        out = run_one(r, n_steps)
        summary[r] = {"recs": out["steps"], "elapsed": time.time() - t0}

    # ---- final-step comparison (the metric that matters) ----
    print("\n\n" + "#" * 92)
    print(f"# RATIO SWEEP SUMMARY -- metrics at the LAST of {n_steps} steps "
          "(ROI crop, step_frac=1.0)")
    print("#" * 92)
    hdr = (
        f"{'ratio':>6} | {'n_faces':>9} | "
        f"{'b0_CD':>9} | {'b0_F1':>7} | {'b1_CD':>9} | {'b1_F1':>7} | "
        f"{'meanF1':>7} | {'sec':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    best = None
    for r in ratios:
        last = summary[r]["recs"][-1]
        b0cd = last.get("bbox_0_crop_chamfer_l1")
        b0f1 = last.get("bbox_0_crop_f1")
        b1cd = last.get("bbox_1_crop_chamfer_l1")
        b1f1 = last.get("bbox_1_crop_f1")
        meanf1 = (
            (b0f1 + b1f1) / 2.0
            if (b0f1 is not None and b1f1 is not None) else float("nan")
        )
        print(
            f"{r:>6.2f} | {last['n_faces']:>9} | "
            f"{(b0cd if b0cd is not None else float('nan')):>9.2e} | "
            f"{(b0f1 if b0f1 is not None else float('nan')):>7.4f} | "
            f"{(b1cd if b1cd is not None else float('nan')):>9.2e} | "
            f"{(b1f1 if b1f1 is not None else float('nan')):>7.4f} | "
            f"{meanf1:>7.4f} | {summary[r]['elapsed']:>6.1f}"
        )
        if not math.isnan(meanf1) and (best is None or meanf1 > best[1]):
            best = (r, meanf1, last["n_faces"])
    if best is not None:
        print("-" * len(hdr))
        print(f"# BEST mean-F1: ratio={best[0]:.2f}  meanF1={best[1]:.4f}  "
              f"n_faces={best[2]}")

    # ---- per-step face growth + F1 trajectory (cost vs gain) ----
    print("\n# per-step n_faces / mean-F1 trajectory")
    for r in ratios:
        recs = summary[r]["recs"]
        traj = " ".join(
            f"[s{rr['step']}:F={rr['n_faces']},"
            f"f1={(((rr.get('bbox_0_crop_f1') or 0)+(rr.get('bbox_1_crop_f1') or 0))/2):.4f}]"
            for rr in recs
        )
        print(f"  ratio {r:.2f}: {traj}")


if __name__ == "__main__":
    main()
