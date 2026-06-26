import time

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.projection_fitter import ProjectionFitter


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
    step_tau: float = 1.0,
    max_sweeps: int = 24,
    gap_margin_tau: float = 0.5,
    max_thickness_tau: float = 30.0,
    thickness_frac: float = 0.5,
    enable_guard: bool = True,
    guard_iters: int = 3,
    bisect_steps: int = 7,
    smooth_lambda: float = 0.0,
    smooth_iters: int = 0,
    rigid_init: bool = True,
    strict_no_intersection: bool = True,
    max_subdivisions: int = 4,
    error_quantile: float = 0.9,
    train_target_samples: int = 2000000,
    eval_samples: int = 2000000,
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
        "method": "projection",
        "step_tau": step_tau,
        "max_sweeps": max_sweeps,
        "gap_margin_tau": gap_margin_tau,
        "max_thickness_tau": max_thickness_tau,
        "thickness_frac": thickness_frac,
        "enable_guard": enable_guard,
        "guard_iters": guard_iters,
        "bisect_steps": bisect_steps,
        "smooth_lambda": smooth_lambda,
        "smooth_iters": smooth_iters,
        "rigid_init": rigid_init,
        "strict_no_intersection": strict_no_intersection,
        "max_subdivisions": max_subdivisions,
        "error_quantile": error_quantile,
        "train_target_samples": train_target_samples,
        "eval_samples": eval_samples,
    }

    fitter = ProjectionFitter(
        device="cuda",
        step_tau=step_tau,
        max_sweeps=max_sweeps,
        gap_margin_tau=gap_margin_tau,
        max_thickness_tau=max_thickness_tau,
        thickness_frac=thickness_frac,
        enable_guard=enable_guard,
        guard_iters=guard_iters,
        bisect_steps=bisect_steps,
        smooth_lambda=smooth_lambda,
        smooth_iters=smooth_iters,
        rigid_init=rigid_init,
        strict_no_intersection=strict_no_intersection,
        max_subdivisions=max_subdivisions,
        error_quantile=error_quantile,
        train_target_samples=train_target_samples,
        eval_samples=eval_samples,
        save_result_folder_path=save_result_folder_path,
    )

    fitter.loadMeshes(source_mesh, target_mesh)

    t = time.time()
    result = fitter.fitAndEvaluate()
    print("[INFO][demo] fit+eval done in", round(time.time() - t, 1), "s")

    print("[INFO][demo] baseline (rigid-init only):")
    for k, v in result["baseline"].items():
        print(f"    {k}: {v}")
    print("[INFO][demo] fitted (projected + refined):")
    for k, v in result["fitted"].items():
        print(f"    {k}: {v}")
    print("[INFO][demo] kept:", result["kept"])
    print("[INFO][demo] refine log:")
    for r in result["refine_log"]:
        print("    ", r)
    print(
        "[INFO][demo] final new self-intersections:",
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
    }
    mesh_path = fitter.saveResult(metrics=metrics_full, config=config)
    print("[INFO][demo] saved mesh to", mesh_path)

    return result


def demo_stepwise(
    data_id: str = "watertight_case1",
    n_steps: int = 4,
    step_frac: float = 0.5,
    stiffness_weight: float = 0.5,
    stiffness_iters: int = 2,
    enable_guard: bool = True,
    guard_iters: int = 8,
    bisect_steps: int = 14,
    min_move_tau: float = 0.02,
    rigid_init: bool = True,
    train_target_samples: int = 2000000,
    eval_samples: int = 500000,
    chamfer_each_step: bool = True,
    save_meshes: bool = True,
    final_gate: bool = True,
    save_result_folder_path: str = "./output/case1_proj_stepwise/",
):
    """Per-step closest-point projection fit (the user-specified loop):
    closest point on target -> step capped to step_frac of the pair distance ->
    stiffness/ARAP shape preservation -> trajectory self-intersection check with
    a UNIFIED (batched, simultaneous) pull-back along ref->current. Runs only the
    first `n_steps`, recording per-step error and mesh.
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

    fitter = ProjectionFitter(
        device="cuda",
        step_frac=step_frac,
        stiffness_weight=stiffness_weight,
        stiffness_iters=stiffness_iters,
        enable_guard=enable_guard,
        guard_iters=guard_iters,
        bisect_steps=bisect_steps,
        min_move_tau=min_move_tau,
        rigid_init=rigid_init,
        max_subdivisions=0,
        train_target_samples=train_target_samples,
        eval_samples=eval_samples,
        save_result_folder_path=save_result_folder_path,
    )
    fitter.loadMeshes(source_mesh, target_mesh)

    t = time.time()
    out = fitter.fitStepwiseProjection(
        n_steps=n_steps,
        save_folder=save_result_folder_path,
        compute_chamfer=True,
        chamfer_each_step=chamfer_each_step,
        save_meshes=save_meshes,
        final_gate=final_gate,
    )
    print("[INFO][demo_stepwise] done in", round(time.time() - t, 1), "s")
    for rec in out["steps"]:
        print("    ", rec)
    return out


if __name__ == "__main__":
    demo()
