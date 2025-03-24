import open3d as o3d
from copy import deepcopy

def renderMeshPair(
    source_mesh: o3d.geometry.TriangleMesh,
    target_mesh: o3d.geometry.TriangleMesh,
) -> bool:
    copied_source_mesh = deepcopy(source_mesh)
    copied_target_mesh = deepcopy(target_mesh)

    copied_source_mesh.paint_uniform_color([0.1,0.1,0.9])
    copied_target_mesh.paint_uniform_color([0.9,0.1,0.1])

    o3d.visualization.draw_geometries([copied_source_mesh, copied_target_mesh])
    return True
