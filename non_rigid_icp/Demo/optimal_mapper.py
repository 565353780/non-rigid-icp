from non_rigid_icp.Module.optimal_mapper import OptimalMapper

def demo():
    source_mesh_file_path = './data/source_test.obj'
    target_mesh_file_path = './data/target_full.obj'
    source_mesh_file_path = './data/SMPL_male.ply'
    target_mesh_file_path = './data/target.ply'
    inner_iter = 50
    outer_iter = 100
    milestones = [50, 80, 100, 110, 120, 130, 140]
    stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]
    laplacian_weight = 0.0
    device = 'cuda'
    save_result_folder_path = 'auto'
    save_log_folder_path = 'auto'

    optimal_mapper = OptimalMapper(
        inner_iter,
        outer_iter,
        milestones,
        stiffness_weights,
        laplacian_weight,
        device,
        save_result_folder_path,
        save_log_folder_path,
    )

    optimal_mapper.loadGTMeshFile(target_mesh_file_path)
    optimal_mapper.loadTemplateMeshFile(source_mesh_file_path)

    optimal_mapper.estimateInitPose()
    optimal_mapper.refineGeometry()

    deformed_mesh = optimal_mapper.toDeformedTemplateMesh()

    optimal_mapper.saveDeformedTemplateMesh('./output/optimal_mapper_mesh.ply')

    print(deformed_mesh)
