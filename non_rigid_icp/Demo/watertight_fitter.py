import time

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


data_dict = {
    # source = watertight mesh (to be deformed), target = original mesh (fitting goal)
    "watertight_case1": [
        "/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_wt_1536.ply",
        "/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_gen.glb",
    ],
    "watertight_case2": [
        "/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case2_wt_1536.ply",
        "/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case2_gen.glb",
    ],
}


def demo(
    data_id: str = "watertight_case1",
    outer_iter: int = 60,
    inner_iter: int = 20,
    train_source_samples: int = 300000,
    train_target_samples: int = 2000000,
    eval_samples: int = 2000000,
    laplacian_weight: float = 200.0,
    point_to_plane_weight: float = 0.0,
    lr: float = 2e-4,
    normal_gate: bool = True,
    enable_self_collision_guard: bool = True,
    collision_weight: float = 200.0,
    collision_margin_tau: float = 0.25,
    collision_broad_tau: float = 1.0,
    collision_refresh_every: int = 4,
    collision_check_every: int = 3,
    enable_sheet_guard: bool = True,
    sheet_gap_tau: float = 1.0,
    sheet_min_margin_tau: float = 0.5,
    sheet_max_thickness_tau: float = 6.0,
    sheet_weight: float = 1000.0,
    enable_inversion_guard: bool = True,
    inversion_weight: float = 20.0,
    enable_trajectory_guard: bool = True,
    trajectory_check_inner_every: int = 5,
    trajectory_active_tau: float = 0.5,
    trajectory_bisect_steps: int = 12,
    trajectory_resolve_iters: int = 4,
    trajectory_final_rounds: int = 24,
    trajectory_dilation_rings: int = 1,
    strict_no_intersection: bool = True,
    max_subdivisions: int = 4,
    refine_iter: int = 25,
    plateau_window: int = 6,
    plateau_rel_tol: float = 3e-3,
    error_quantile: float = 0.9,
    save_result_folder_path: str = "auto",
):
    source_mesh_file_path, target_mesh_file_path = data_dict[data_id]

    print("[INFO][demo] loading source:", source_mesh_file_path)
    t = time.time()
    source_mesh = Mesh(source_mesh_file_path)
    print("[INFO][demo] source loaded in", round(time.time() - t, 1), "s")

    print("[INFO][demo] loading target:", target_mesh_file_path)
    t = time.time()
    target_mesh = Mesh(target_mesh_file_path)
    print("[INFO][demo] target loaded in", round(time.time() - t, 1), "s")

    config = {
        "data_id": data_id,
        "outer_iter": outer_iter,
        "inner_iter": inner_iter,
        "train_source_samples": train_source_samples,
        "train_target_samples": train_target_samples,
        "eval_samples": eval_samples,
        "laplacian_weight": laplacian_weight,
        "point_to_plane_weight": point_to_plane_weight,
        "lr": lr,
        "normal_gate": normal_gate,
        "enable_self_collision_guard": enable_self_collision_guard,
        "collision_weight": collision_weight,
        "collision_margin_tau": collision_margin_tau,
        "collision_broad_tau": collision_broad_tau,
        "collision_refresh_every": collision_refresh_every,
        "collision_check_every": collision_check_every,
        "enable_sheet_guard": enable_sheet_guard,
        "sheet_gap_tau": sheet_gap_tau,
        "sheet_min_margin_tau": sheet_min_margin_tau,
        "sheet_max_thickness_tau": sheet_max_thickness_tau,
        "sheet_weight": sheet_weight,
        "enable_inversion_guard": enable_inversion_guard,
        "inversion_weight": inversion_weight,
        "enable_trajectory_guard": enable_trajectory_guard,
        "trajectory_check_inner_every": trajectory_check_inner_every,
        "trajectory_active_tau": trajectory_active_tau,
        "trajectory_bisect_steps": trajectory_bisect_steps,
        "trajectory_resolve_iters": trajectory_resolve_iters,
        "trajectory_final_rounds": trajectory_final_rounds,
        "trajectory_dilation_rings": trajectory_dilation_rings,
        "strict_no_intersection": strict_no_intersection,
        "max_subdivisions": max_subdivisions,
        "refine_iter": refine_iter,
        "plateau_window": plateau_window,
        "plateau_rel_tol": plateau_rel_tol,
        "error_quantile": error_quantile,
    }

    fitter = WatertightFitter(
        device="cuda",
        outer_iter=outer_iter,
        inner_iter=inner_iter,
        lr=lr,
        normal_gate=normal_gate,
        train_source_samples=train_source_samples,
        train_target_samples=train_target_samples,
        eval_samples=eval_samples,
        laplacian_weight=laplacian_weight,
        point_to_plane_weight=point_to_plane_weight,
        enable_self_collision_guard=enable_self_collision_guard,
        collision_weight=collision_weight,
        collision_margin_tau=collision_margin_tau,
        collision_broad_tau=collision_broad_tau,
        collision_refresh_every=collision_refresh_every,
        collision_check_every=collision_check_every,
        enable_sheet_guard=enable_sheet_guard,
        sheet_gap_tau=sheet_gap_tau,
        sheet_min_margin_tau=sheet_min_margin_tau,
        sheet_max_thickness_tau=sheet_max_thickness_tau,
        sheet_weight=sheet_weight,
        enable_inversion_guard=enable_inversion_guard,
        inversion_weight=inversion_weight,
        enable_trajectory_guard=enable_trajectory_guard,
        trajectory_check_inner_every=trajectory_check_inner_every,
        trajectory_active_tau=trajectory_active_tau,
        trajectory_bisect_steps=trajectory_bisect_steps,
        trajectory_resolve_iters=trajectory_resolve_iters,
        trajectory_final_rounds=trajectory_final_rounds,
        trajectory_dilation_rings=trajectory_dilation_rings,
        strict_no_intersection=strict_no_intersection,
        max_subdivisions=max_subdivisions,
        refine_iter=refine_iter,
        plateau_window=plateau_window,
        plateau_rel_tol=plateau_rel_tol,
        error_quantile=error_quantile,
        save_result_folder_path=save_result_folder_path,
    )

    fitter.loadMeshes(source_mesh, target_mesh)

    t = time.time()
    result = fitter.fitAndEvaluate()
    print("[INFO][demo] fit+eval done in", round(time.time() - t, 1), "s")

    print("[INFO][demo] baseline (rigid-init only):")
    for k, v in result["baseline"].items():
        print(f"    {k}: {v}")
    print("[INFO][demo] fitted (deformed + refined):")
    for k, v in result["fitted"].items():
        print(f"    {k}: {v}")
    print("[INFO][demo] kept:", result["kept"])
    print("[INFO][demo] refine log:")
    for r in result["refine_log"]:
        print("    ", r)
    print(
        "[INFO][demo] trajectory crossings:",
        "vertices=", result.get("trajectory_crossing_vertices"),
        "pairs=", result.get("trajectory_crossing_pairs"),
        "faces=", result.get("trajectory_crossing_faces"),
        "| trajectory_free:", result.get("trajectory_self_intersection_free"),
    )
    print(
        "[INFO][demo] triangle-triangle new self-intersections (supplementary):",
        result["final_new_self_intersections"],
        "| self_intersection_free:",
        result.get("self_intersection_free"),
    )

    metrics_full = {
        "kept": result["kept"],
        "baseline": result["baseline"],
        "fitted": result["fitted"],
        "refine_log": result["refine_log"],
        "final_new_self_intersections": result["final_new_self_intersections"],
        "self_intersection_free": result.get("self_intersection_free"),
        "trajectory_crossing_vertices": result.get("trajectory_crossing_vertices"),
        "trajectory_crossing_pairs": result.get("trajectory_crossing_pairs"),
        "trajectory_crossing_faces": result.get("trajectory_crossing_faces"),
        "trajectory_self_intersection_free": result.get(
            "trajectory_self_intersection_free"
        ),
    }
    mesh_path = fitter.saveResult(metrics=metrics_full, config=config)
    print("[INFO][demo] saved mesh to", mesh_path)

    return result


def demo_stepwise(
    data_id: str = "watertight_case1",
    n_steps: int = 4,
    inner_per_step: int = 1,
    # bigger step + lighter Laplacian so each recorded step visibly hugs the
    # target (the v4 fit defaults lr=2e-4/lap=200 barely move a 14M-vert mesh
    # per step, which is why the error looked flat).
    lr: float = 1e-3,
    laplacian_weight: float = 30.0,
    # freeze vertices already within this many tau of the target so the data
    # term only moves the genuinely high-error region (case1 is already ~1.2tau
    # after rigid ICP, so optimizing everything just thrashes fitted layers).
    error_gate_tau: float = 1.0,
    normal_gate: bool = True,
    train_target_samples: int = 2000000,
    eval_samples: int = 500000,
    trajectory_resolve_iters: int = 6,
    trajectory_active_tau: float = 0.5,
    trajectory_bisect_steps: int = 12,
    trajectory_check_inner_every: int = 1,
    compute_chamfer: bool = True,
    chamfer_each_step: bool = False,
    save_meshes: bool = True,
    save_result_folder_path: str = "./output/case1_stepwise/",
):
    """Per-step debug run: take the first `n_steps` optimization steps, each one
    pulling the mesh closer to the target then immediately checking + locally
    repairing trajectory self-intersections, recording per-step error and mesh.
    The full Chamfer is computed once at the end by default (per-step Chamfer is
    expensive); set chamfer_each_step=True to record it every step.
    """
    source_mesh_file_path, target_mesh_file_path = data_dict[data_id]

    print("[INFO][demo_stepwise] loading source:", source_mesh_file_path)
    t = time.time()
    source_mesh = Mesh(source_mesh_file_path)
    print("[INFO][demo_stepwise] source loaded in", round(time.time() - t, 1), "s")

    print("[INFO][demo_stepwise] loading target:", target_mesh_file_path)
    t = time.time()
    target_mesh = Mesh(target_mesh_file_path)
    print("[INFO][demo_stepwise] target loaded in", round(time.time() - t, 1), "s")

    fitter = WatertightFitter(
        device="cuda",
        lr=lr,
        laplacian_weight=laplacian_weight,
        normal_gate=normal_gate,
        train_target_samples=train_target_samples,
        eval_samples=eval_samples,
        enable_trajectory_guard=True,
        trajectory_resolve_iters=trajectory_resolve_iters,
        trajectory_active_tau=trajectory_active_tau,
        trajectory_bisect_steps=trajectory_bisect_steps,
        trajectory_check_inner_every=trajectory_check_inner_every,
        # no subdivision in the stepwise diagnostic
        max_subdivisions=0,
        save_result_folder_path=save_result_folder_path,
    )
    fitter.loadMeshes(source_mesh, target_mesh)

    t = time.time()
    out = fitter.fitStepwise(
        n_steps=n_steps,
        inner_per_step=inner_per_step,
        error_gate_tau=error_gate_tau,
        save_folder=save_result_folder_path,
        compute_chamfer=compute_chamfer,
        chamfer_each_step=chamfer_each_step,
        save_meshes=save_meshes,
    )
    print("[INFO][demo_stepwise] done in", round(time.time() - t, 1), "s")
    for rec in out["steps"]:
        print("    ", rec)
    return out


def demo_stepwise_clamped(
    data_id: str = "watertight_case1",
    # The per-vertex cap step_frac*d_i makes the mean residual decay as
    # (1-step_frac)^n -- step_frac is literally the fraction of the way each
    # vertex jumps toward its exact closest point on the target. A full sweep on
    # the ROI-cropped case1 (Test/exp_step_frac_sweep.py, step_frac in
    # [0.1..2.0] with gd_lr=2*step_frac so the cap, not the raw GD step, is the
    # limiter) located the stability ceiling exactly:
    #   * step_frac <= 1.0 : STABLE -- 0 self-intersection repairs, F1 climbs
    #       monotonically to its ceiling (~0.818/0.727), residual -> ~0.
    #   * step_frac == 1.0 : OPTIMUM -- one-shot projection: residual collapses
    #       0.221tau -> 0.000tau in a SINGLE step, F1 maxes immediately.
    #   * step_frac == 1.5 : UNSTABLE -- overshoot pierces faces, 1-11 pull-back
    #       repairs EVERY step, never fully settles.
    #   * step_frac == 2.0 : BROKEN -- 134 repairs, residual stuck ~0.215tau
    #       (oscillates), F1 degrades below the start.
    # So step_frac=1.0 (gd_lr=2.0 so the raw step can reach the cap) is the
    # largest stable step and fits in ~1-3 base steps; drop to 0.8 for a
    # one-step safety margin on shells the ROI crop under-represents.
    n_steps: int = 40,
    step_frac: float = 1.0,
    gd_lr: float = 2.0,
    laplacian_weight: float = 30.0,
    n_bisect: int = 16,
    resolve_iters: int = 8,
    pullback_min_move_tau: float = 0.0,
    trajectory_seg_chunk: int = 200000,
    # SAG-DRIVEN adaptive subdivision. At step_frac=1.0 every vertex lands on the
    # target in ~1 step, so the residual plateaus near zero almost immediately;
    # the plateau then just gates "vertices settled -> look for sagging faces".
    # A face is split ONLY when its interior tents off the target beyond its
    # corners (sag>refine_sag_mult*tau and centroid out of tolerance) -- the sole
    # split that lowers surface-to-surface error -- so the count grows minimally
    # and stops automatically once nothing sags. Small window/patience let a
    # round fire every ~2 steps; max_subdivisions is the level cap (each level
    # quarters a sagging face), generous here so 40 steps can fully resolve.
    max_subdivisions: int = 12,
    plateau_window: int = 1,
    plateau_rel_tol: float = 5e-3,
    plateau_patience: int = 1,
    converge_abs_tau: float = 0.05,
    # sag-based localizer thresholds. 2.0 (split only faces tented >2tau, where a
    # diagnostic on case1 found ZERO after projection -> minimal/no subdivision,
    # the fewest-faces optimum); >=1 would chase the target's ~tau discretization
    # noise and blow up the face count without improving F1.
    refine_sag_mult: float = 2.0,
    refine_centroid_mult: float = 2.0,
    refine_sag_quantile: float = 0.0,
    # legacy vertex-error localizer knobs (unused while refine_sag_mult>0).
    error_mult: float = 1.0,
    error_quantile: float = 0.9,
    max_refine_faces: int = 1500000,
    # unoptimizable-vertex state machine (drop unreachable thin-shell vertices
    # from the data loss + skip them in adaptive subdivision)
    unopt_error_tau: float = 1.0,
    unopt_min_intended_move_tau: float = 0.02,
    unopt_min_actual_move_tau: float = 0.004,
    unopt_min_progress_ratio: float = 0.1,
    unopt_block_patience: int = 3,
    local_drop_tau: float = 0.02,
    max_blocked_vertex_ratio: float = 0.5,
    refine_cooldown: int = 1,
    # region-restricted (bbox) evaluation, in the de-normalized / original frame.
    # Two regions of interest are tracked + saved independently (debug/<name>/).
    eval_bboxes=(
        {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
        {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
    ),
    eval_bbox_mode: str = "all",
    crop_eval_samples: int = 300000,
    crop_eval: bool = True,
    full_chamfer_each_step: bool = False,
    train_target_samples: int = 2000000,
    eval_samples: int = 500000,
    save_full_each_step: bool = False,
    save_result_folder_path: str = "./output/case1_stepwise_clamped/",
):
    """Gradient-descent stepwise fit with a per-vertex step cap + adaptive
    subdivision (user spec): each step moves every vertex along the
    data+Laplacian gradient toward its exact closest point on the target, but no
    more than step_frac * d_i, then checks the ref->current segment against ANY
    face and pulls every offending vertex back along that segment to the largest
    crossing-free position. Near convergence (plateau) the high-error region is
    locally subdivided -- on both the deformed mesh AND the rest reference in
    lock-step. Evaluation is restricted to the eval bbox and the crops are saved
    under <save_folder>/debug/. Runs the first `n_steps`.
    """
    source_mesh_file_path, target_mesh_file_path = data_dict[data_id]

    print("[INFO][demo_stepwise_clamped] loading source:", source_mesh_file_path)
    t = time.time()
    source_mesh = Mesh(source_mesh_file_path)
    print("[INFO][demo_stepwise_clamped] source loaded in", round(time.time() - t, 1), "s")

    print("[INFO][demo_stepwise_clamped] loading target:", target_mesh_file_path)
    t = time.time()
    target_mesh = Mesh(target_mesh_file_path)
    print("[INFO][demo_stepwise_clamped] target loaded in", round(time.time() - t, 1), "s")

    fitter = WatertightFitter(
        device="cuda",
        laplacian_weight=laplacian_weight,
        normal_gate=False,
        train_target_samples=train_target_samples,
        eval_samples=eval_samples,
        # the clamped path enforces no-self-intersection purely via the
        # trajectory pull-back, so the collision/sheet/inversion guards and the
        # (expensive, full-mesh) baseline self-intersection scan are not needed.
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        trajectory_seg_chunk=trajectory_seg_chunk,
        max_subdivisions=max_subdivisions,
        plateau_window=plateau_window,
        plateau_rel_tol=plateau_rel_tol,
        plateau_patience=plateau_patience,
        error_mult=error_mult,
        error_quantile=error_quantile,
        max_refine_faces=max_refine_faces,
        refine_sag_mult=refine_sag_mult,
        refine_centroid_mult=refine_centroid_mult,
        refine_sag_quantile=refine_sag_quantile,
        unopt_error_tau=unopt_error_tau,
        unopt_min_intended_move_tau=unopt_min_intended_move_tau,
        unopt_min_actual_move_tau=unopt_min_actual_move_tau,
        unopt_min_progress_ratio=unopt_min_progress_ratio,
        unopt_block_patience=unopt_block_patience,
        local_drop_tau=local_drop_tau,
        max_blocked_vertex_ratio=max_blocked_vertex_ratio,
        refine_cooldown=refine_cooldown,
        eval_bboxes=eval_bboxes,
        eval_bbox_mode=eval_bbox_mode,
        crop_eval_samples=crop_eval_samples,
        save_result_folder_path=save_result_folder_path,
    )
    fitter.loadMeshes(source_mesh, target_mesh)

    t = time.time()
    out = fitter.fitStepwiseClamped(
        n_steps=n_steps,
        step_frac=step_frac,
        gd_lr=gd_lr,
        lap_w=laplacian_weight,
        n_bisect=n_bisect,
        resolve_iters=resolve_iters,
        pullback_min_move_tau=pullback_min_move_tau,
        max_subdivisions=max_subdivisions,
        converge_abs_tau=converge_abs_tau,
        save_folder=save_result_folder_path,
        compute_chamfer=True,
        crop_eval=crop_eval,
        full_chamfer_each_step=full_chamfer_each_step,
        save_full_each_step=save_full_each_step,
    )
    print("[INFO][demo_stepwise_clamped] done in", round(time.time() - t, 1), "s")
    for rec in out["steps"]:
        print("    ", rec)
    return out


if __name__ == "__main__":
    demo()
