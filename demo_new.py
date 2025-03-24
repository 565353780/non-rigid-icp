import os
import torch
import numpy as np
import open3d as o3d
from copy import deepcopy

from non_rigid_icp.Method.trans import toNormalizeTransform, transMesh
from non_rigid_icp.Method.registration import registration_mesh2mesh

def pose_registration(
    tmpl_mesh_path: str,
    target_mesh_path: str,
    device: str='cpu',
) -> o3d.geometry.TriangleMesh:
    template_mesh = o3d.io.read_triangle_mesh(tmpl_mesh_path)
    target_mesh = o3d.io.read_triangle_mesh(target_mesh_path)

    template_points = np.asarray(template_mesh.vertices)
    target_points = np.asarray(target_mesh.vertices)

    template_center, template_scale = toNormalizeTransform(template_points)
    target_center, target_scale = toNormalizeTransform(target_points)

    transMesh(template_mesh, template_center, template_scale)
    transMesh(target_mesh, target_center, target_scale)

    template_mesh.compute_vertex_normals()
    target_mesh.compute_vertex_normals()

    registered_mesh = registration_mesh2mesh(template_mesh,
                                             target_mesh,
                                             device=device)
    assert isinstance(registered_mesh, o3d.geometry.TriangleMesh)

    registered_mesh = deepcopy(template_mesh)
    transMesh(registered_mesh, target_center, target_scale, True)

    registered_mesh.compute_vertex_normals()

    return registered_mesh

if __name__ == "__main__":
    template_mesh_path = './data/SMPL_male.ply'
    target_mesh_path = './data/target.ply'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    deformed_mesh = pose_registration(template_mesh_path, target_mesh_path, device)

    os.makedirs('./output/', exist_ok=True)
    o3d.io.write_triangle_mesh('./output/registered_mesh.ply', deformed_mesh)
