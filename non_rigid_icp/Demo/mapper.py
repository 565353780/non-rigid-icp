from non_rigid_icp.Module.mapper import Mapper

def demo():
    source_mesh_file_path = './data/source_test.obj'
    target_mesh_file_path = './data/target_full.obj'
    source_mesh_file_path = './data/SMPL_male.ply'
    target_mesh_file_path = './data/target.ply'
    render = True

    mapper = Mapper()
    deformed_mesh = mapper.mapMesh(
        source_mesh_file_path,
        target_mesh_file_path,
        render
    )

    print(deformed_mesh)
