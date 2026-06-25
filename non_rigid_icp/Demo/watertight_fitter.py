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
    }

    fitter = WatertightFitter(
        device="cuda",
        outer_iter=outer_iter,
        inner_iter=inner_iter,
        lr=lr,
        train_source_samples=train_source_samples,
        train_target_samples=train_target_samples,
        eval_samples=eval_samples,
        laplacian_weight=laplacian_weight,
        point_to_plane_weight=point_to_plane_weight,
        save_result_folder_path=save_result_folder_path,
    )

    fitter.loadMeshes(source_mesh, target_mesh)

    t = time.time()
    result = fitter.fitAndEvaluate()
    print("[INFO][demo] fit+eval done in", round(time.time() - t, 1), "s")

    print("[INFO][demo] baseline (rigid-init only):")
    for k, v in result["baseline"].items():
        print(f"    {k}: {v}")
    print("[INFO][demo] fitted (deformed):")
    for k, v in result["fitted"].items():
        print(f"    {k}: {v}")
    print("[INFO][demo] kept:", result["kept"])

    metrics = result["metrics"]
    metrics_full = {
        "kept": result["kept"],
        "baseline": result["baseline"],
        "fitted": result["fitted"],
    }
    mesh_path = fitter.saveResult(metrics=metrics_full, config=config)
    print("[INFO][demo] saved mesh to", mesh_path)

    return result


if __name__ == "__main__":
    demo()
