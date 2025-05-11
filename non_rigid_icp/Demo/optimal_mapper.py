from non_rigid_icp.Module.optimal_mapper import OptimalMapper

data_dict = {
    'test': [
        './data/source_test.obj',
        './data/target_full.obj',
    ],
    'SMPL': [
        './data/SMPL_male.ply',
        './data/target.ply',
    ],
    'airplane_head': [
        '/home/chli/chLi/Dataset/AMCAX/mesh-fitting/AMCAX_airplane_head_template.ply',
        '/home/chli/chLi/Dataset/AMCAX/mesh-fitting/AMCAX_airplane_head_target.ply',
    ],
}

def demo():
    data_id = 'airplane_head'

    source_mesh_file_path, target_mesh_file_path = data_dict[data_id]
    inner_iter = 50
    outer_iter = 200
    milestones = [50, 80, 100, 110, 120, 130, 140, 150]
    stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2, 0]
    laplacian_weight = 1.0
    device = 'cuda'
    save_result_folder_path = 'auto'
    save_log_folder_path = 'auto'
    render = True

    optimal_mapper = OptimalMapper(
        inner_iter,
        outer_iter,
        milestones,
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

    optimal_mapper.saveDeformedTemplateMesh(optimal_mapper.save_result_folder_path + 'optimal_mapper_mesh.ply')

    print(deformed_mesh)
