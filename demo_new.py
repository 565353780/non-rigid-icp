import os
import torch
import trimesh
import numpy as np
import open3d as o3d

from non_rigid_icp.Method.utils import toNormalizeTransform
from non_rigid_icp.Method.registration import registration_mesh2mesh

def pose_registration(
    tmpl_mesh_path: str,
    target_mesh_path: str,
    device: str='cpu',
) -> o3d.geometry.TriangleMesh:
    tri_mesh = trimesh.load_mesh(tmpl_mesh_path, process=True)
    target_mesh = o3d.io.read_triangle_mesh(target_mesh_path,
                                            enable_post_processing=True)

    template_points = tri_mesh.vertices
    target_points = np.asarray(target_mesh.vertices)

    template_center, template_scale = toNormalizeTransform(template_points)
    target_center, target_scale = toNormalizeTransform(target_points)

    normalized_template_points = (template_points - template_center) * template_scale
    normalized_target_points = (target_points - target_center) * target_scale

    template_mesh = o3d.geometry.TriangleMesh()
    template_mesh.vertices = o3d.utility.Vector3dVector(normalized_template_points)
    template_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    template_mesh.compute_vertex_normals()

    target_mesh.vertices = o3d.utility.Vector3dVector(normalized_target_points)
    target_mesh.compute_vertex_normals()

    registered_mesh = registration_mesh2mesh(template_mesh,
                                             target_mesh,
                                             device=device)
    assert isinstance(registered_mesh, o3d.geometry.TriangleMesh)

    registered_points = np.asarray(registered_mesh.vertices)
    registered_points = registered_points / target_scale + target_center

    registered_mesh.vertices = o3d.utility.Vector3dVector(registered_points)
    registered_mesh.compute_vertex_normals()

    return registered_mesh

if __name__ == "__main__":
    template_mesh_path = './data/SMPL_male.ply'
    target_mesh_path = './data/target.ply'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    deformed_mesh = pose_registration(template_mesh_path, target_mesh_path, device)

    os.makedirs('./output/', exist_ok=True)
    o3d.io.write_triangle_mesh('./output/registered_mesh.ply', deformed_mesh)
