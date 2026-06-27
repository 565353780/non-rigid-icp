"""Sweep the per-step clamp fraction (step_frac) over the FIRST few steps to find
the largest step that still finishes the fit in ~10 steps without exploding the
trajectory pull-back / clamp-clipping.

Fast validation: the FULL source+target are loaded and rigidly aligned, then the
optional pre-fit crop (WatertightFitter(prefit_crop_bboxes=...)) shrinks BOTH
meshes -- in lock-step, with the identical boxes -- to just the regions of
interest, so each kept region is bit-for-bit what the full-mesh run would
optimize while the run is orders of magnitude faster. Drop `prefit_crop_bboxes`
to reproduce on the full mesh.

Rationale (first principles): with a per-vertex cap of step_frac * d_i, the mean
residual obeys roughly resid_n ~ (1 - eff)^n * resid_0, where `eff` is the
EFFECTIVE per-step decay -- <= step_frac because the Laplacian smooths the move
before the cap and the trajectory pull-back drags some vertices back. We measure
`eff` from the first steps for each candidate and extrapolate how many steps it
takes to reach a target residual (in tau). Subdivision is DISABLED so we isolate
the pure step-size behaviour.
"""

import math
import time

from non_rigid_icp.Demo.watertight_fitter import data_dict
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


# two regions of interest (original-target frame). Used BOTH as the pre-fit crop
# (shrink the 14M mesh to just these regions) AND as the eval bboxes -- so the
# validated region is exactly the evaluated region.
ROI_BBOXES = (
    {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
    {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
)


def run_one(step_frac: float, n_steps: int, gd_lr: float, lap_w: float):
    # The raw data GD step magnitude is gd_lr * fit_weight * d_i = 0.5*d_i at the
    # default gd_lr=0.5, so the per-vertex cap step_frac*d_i only BINDS while
    # step_frac < 0.5; above that the raw 0.5*d_i step is the real limiter. To
    # genuinely probe step_frac >= 0.5 we scale gd_lr = 2*step_frac so the raw
    # step always equals the cap and step_frac is the true move fraction.
    gd_lr = max(gd_lr, 2.0 * step_frac)
    src_path, tgt_path = data_dict["watertight_case1"]
    source_mesh = Mesh(src_path)
    target_mesh = Mesh(tgt_path)

    fitter = WatertightFitter(
        device="cuda",
        laplacian_weight=lap_w,
        normal_gate=False,
        train_target_samples=2000000,
        eval_samples=500000,
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        trajectory_seg_chunk=200000,
        # isolate pure step-size behaviour: NO subdivision, NO unopt dropping
        max_subdivisions=0,
        # FAST validation: crop source+target to the ROIs before optimizing.
        # `centroid` keeps boundary-straddling faces so the cropped surface is
        # closed enough for a faithful local fit. (Drop this arg to run full.)
        prefit_crop_bboxes=ROI_BBOXES,
        prefit_crop_mode="centroid",
        eval_bboxes=ROI_BBOXES,
        save_result_folder_path=f"./output/exp_step_frac/frac_{step_frac:.2f}/",
    )
    fitter.loadMeshes(source_mesh, target_mesh)

    out = fitter.fitStepwiseClamped(
        n_steps=n_steps,
        step_frac=step_frac,
        gd_lr=gd_lr,
        lap_w=lap_w,
        max_subdivisions=0,
        converge_abs_tau=0.0,        # never auto-subdivide
        compute_chamfer=False,        # skip the expensive final full chamfer
        crop_eval=True,
        full_chamfer_each_step=False,
        save_full_each_step=False,
    )
    return out


def main():
    # push past 0.5 to find the stable maximum: at <=0.5 clip%=0 and 0
    # self-intersection repairs (no binding limit yet). step_frac>1 OVERSHOOTS
    # the closest point, so it is the regime where the trajectory pull-back /
    # clamp interaction is expected to start clipping or repairing -- that onset
    # marks the stability ceiling. Run a few extra steps so any post-"convergence"
    # oscillation / self-intersection from an over-large step is visible.
    import os
    fracs_env = os.environ.get("SWEEP_FRACS")
    if fracs_env:
        fracs = [float(x) for x in fracs_env.split(",")]
    else:
        fracs = [0.6, 0.8, 1.0, 1.5, 2.0]
    n_steps = int(os.environ.get("SWEEP_NSTEPS", "6"))
    gd_lr = 0.5
    lap_w = 30.0

    summary = {}
    for f in fracs:
        print("\n" + "=" * 70)
        print(f"[EXP] step_frac = {f}")
        print("=" * 70)
        t0 = time.time()
        out = run_one(f, n_steps, gd_lr, lap_w)
        summary[f] = {"recs": out["steps"], "elapsed": time.time() - t0}

    # ---- per-step comparison table ----
    print("\n\n" + "#" * 84)
    print("# STEP_FRAC SWEEP SUMMARY (first {} steps, no subdivision, ROI crop)".format(n_steps))
    print("#" * 84)
    header = (
        f"{'frac':>5} | {'step':>4} | {'resid_tau':>9} | "
        f"{'intended':>8} | {'actual':>8} | {'clip%':>6} | "
        f"{'repaired':>8} | {'b0_f1':>6} | {'b1_f1':>6} | {'sec':>5}"
    )
    print(header)
    print("-" * len(header))
    for f in fracs:
        for r in summary[f]["recs"]:
            im = r["mean_step_move_tau"]
            am = r["mean_actual_move_tau"]
            clip = (1.0 - am / im) * 100.0 if im > 1e-9 else 0.0
            b0 = r.get("bbox_0_crop_f1", float("nan"))
            b1 = r.get("bbox_1_crop_f1", float("nan"))
            print(
                f"{f:>5.2f} | {r['step']:>4} | {r['fit_residual_mean_tau']:>9.3f} | "
                f"{im:>8.4f} | {am:>8.4f} | {clip:>6.1f} | "
                f"{r['trajectory_repaired_vertices']:>8} | {b0:>6.3f} | "
                f"{b1:>6.3f} | {r['seconds']:>5.1f}"
            )
        print("-" * len(header))

    # ---- effective decay + step-count extrapolation ----
    print("\n# EFFECTIVE DECAY + steps-to-converge extrapolation")
    print(f"{'frac':>5} | {'r0':>7} | {'r_last':>7} | {'eff/step':>8} | "
          f"{'steps->1tau':>11} | {'steps->2tau':>11}")
    print("-" * 64)
    for f in fracs:
        recs = summary[f]["recs"]
        r0 = recs[0]["fit_residual_mean_tau"]
        r_last = recs[-1]["fit_residual_mean_tau"]
        n = len(recs)
        if r0 > 1e-9 and r_last > 1e-9 and n > 1:
            eff = 1.0 - (r_last / r0) ** (1.0 / (n - 1))
        else:
            eff = float("nan")

        def steps_to(target):
            if not (0 < eff < 1) or r0 <= target:
                return float("nan")
            return math.log(target / r0) / math.log(1.0 - eff) + 1.0

        print(
            f"{f:>5.2f} | {r0:>7.3f} | {r_last:>7.3f} | {eff:>8.3f} | "
            f"{steps_to(1.0):>11.1f} | {steps_to(2.0):>11.1f}"
        )


if __name__ == "__main__":
    main()
