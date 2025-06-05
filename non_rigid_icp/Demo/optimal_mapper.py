import numpy as np

from non_rigid_icp.Data.mesh import Mesh
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

    source_mesh_file_path, target_mesh_file_path = data_dict[data_id]
    inner_iter = 50
    outer_iter = 200
    milestones = np.arange(10, outer_iter, 4)
    masked_dist_thresh = 0.04
    masked_dist_thresh = float("inf")
    masked_dist_weight = 1.0
    stiffness_weights = 64 * 0.8 ** np.arange(milestones.shape[0] + 1)
    laplacian_weight = 1.0
    device = "cuda"
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    render = True

    print("milestones:", milestones)
    print("stiffness_weights:", stiffness_weights)

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

    source_mesh = Mesh(source_mesh_file_path)
    source_mesh.normalize()
    optimal_mapper.loadTemplateMesh(source_mesh.vertices, source_mesh.triangles)

    target_mesh = Mesh(target_mesh_file_path)
    target_mesh.normalize()
    o3d_mesh = target_mesh.toO3DMesh()
    # fps_pcd = o3d_mesh.sample_points_uniformly(4 * target_mesh.vertices.shape[0])
    # target_points = np.asarray(fps_pcd.points)
    optimal_mapper.addTargetPointsConstraint(target_mesh.vertices)

    group_attr = source_mesh.attributes["vertex_Group"].astype(np.int64)

    grouped_vertex_idxs = np.where(group_attr == 1)[0]
    optimal_mapper.addVertexGroupConstraint(0, grouped_vertex_idxs)

    fixed_vertex_idxs = np.where(group_attr == 2)[0]
    fixed_target_positions = target_mesh.vertices[fixed_vertex_idxs]
    optimal_mapper.addFixedVertexConstraint(fixed_vertex_idxs, fixed_target_positions)

    optimal_mapper.map()

    deformed_mesh = optimal_mapper.toDeformedTemplateMesh()

    deformed_mesh.transform(
        target_mesh.norm_center, target_mesh.norm_scale, is_inverse=True
    )

    deformed_mesh.save(
        optimal_mapper.save_result_folder_path + "optimal_mapper_mesh.ply"
    )

    print(deformed_mesh)
