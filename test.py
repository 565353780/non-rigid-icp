import torch
import numpy as np

from non_rigid_icp.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from non_rigid_icp.Data.mesh import Mesh

target_mesh = Mesh(
    "/home/chli/chLi/Dataset/AMCAX/mesh-fitting/AMCAX_airplane_head_target.ply"
)
fitting_mesh = Mesh("./output/20250601_15:18:47/optimal_mapper_mesh.ply")

target_o3d_mesh = target_mesh.toO3DMesh()
fitting_o3d_mesh = fitting_mesh.toO3DMesh()

target_fps_pcd = target_o3d_mesh.sample_points_uniformly(
    4 * target_mesh.vertices.shape[0]
)
fitting_fps_pcd = fitting_o3d_mesh.sample_points_uniformly(
    4 * fitting_mesh.vertices.shape[0]
)

target_pts = (
    torch.from_numpy(np.asarray(target_fps_pcd.points))
    .unsqueeze(0)
    .to("cuda", dtype=torch.float32)
)
fitting_pts = (
    torch.from_numpy(np.asarray(fitting_fps_pcd.points))
    .unsqueeze(0)
    .to("cuda", dtype=torch.float32)
)

dists1, dists2 = chamfer_3DDist()(fitting_pts, target_pts)[:2]

fit_error = torch.mean(dists1)
cov_error = torch.mean(dists2)

print(fit_error, torch.sqrt(fit_error))
print(cov_error, torch.sqrt(cov_error))
