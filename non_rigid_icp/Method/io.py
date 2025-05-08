import os
import open3d as o3d
from typing import Union


def loadMeshFile(mesh_file_path: str) -> Union[o3d.geometry.TriangleMesh, None]:
    if not os.path.exists(mesh_file_path):
        print("[ERROR][io::loadMeshFile]")
        print("\t mesh file not exist!")
        print("\t mesh_file_path:", mesh_file_path)
        return None

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh
