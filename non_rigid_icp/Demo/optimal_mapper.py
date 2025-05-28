import trimesh
import numpy as np

from non_rigid_icp.Module.optimal_mapper import OptimalMapper


data_dict = {
    "test": [
        "./data/source_test.obj",
        "./data/target_full.obj",
    ],
    "SMPL": [
        "./data/SMPL_male.ply",
        "./data/target.ply",
    ],
    "airplane_head": [
        "/home/chli/chLi/Dataset/AMCAX/mesh-fitting/AMCAX_airplane_head_template.ply",
        "/home/chli/chLi/Dataset/AMCAX/mesh-fitting/AMCAX_airplane_head_target.ply",
    ],
}


def demo():
    data_id = "airplane_head"

    assert_single_manifold = False

    source_mesh_file_path, target_mesh_file_path = data_dict[data_id]
    s_mesh = trimesh.load(source_mesh_file_path)
    is_connected = len(s_mesh.split()) == 1
    if not is_connected:
        print("[WARN][optimal_mapper::demo]")
        print("\t source mesh is not single manifold!")
        if assert_single_manifold:
            return False

    source_mesh_file_path, target_mesh_file_path = data_dict[data_id]
    t_mesh = trimesh.load(source_mesh_file_path)
    is_connected = len(t_mesh.split()) == 1
    if not is_connected:
        print("[WARN][optimal_mapper::demo]")
        print("\t target mesh is not single manifold!")
        if assert_single_manifold:
            return False

    inner_iter = 50
    outer_iter = 400
    milestones = np.arange(10, outer_iter, 4)
    masked_dist_thresh = float("inf")
    masked_dist_weight = 1.0
    stiffness_weights = 64 * 0.8 ** np.arange(milestones.shape[0] + 1)
    laplacian_weight = 1.0
    device = "cuda"
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    render = True

    optimal_mapper = OptimalMapper(
        inner_iter,
        outer_iter,
        milestones,
        masked_dist_thresh,
        masked_dist_weight,
        stiffness_weights,
        laplacian_weight,
        device,
        save_result_folder_path,
        save_log_folder_path,
        render,
    )

    optimal_mapper.loadGTMeshFile(target_mesh_file_path)
    optimal_mapper.loadTemplateMeshFile(source_mesh_file_path)

    optimal_mapper.estimateInitPose()
    optimal_mapper.refineGeometry()

    deformed_mesh = optimal_mapper.toDeformedTemplateMesh()

    optimal_mapper.saveDeformedTemplateMesh(
        optimal_mapper.save_result_folder_path + "optimal_mapper_mesh.ply"
    )

    print(deformed_mesh)
