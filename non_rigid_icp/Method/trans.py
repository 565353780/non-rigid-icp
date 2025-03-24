import torch
import trimesh
import numpy as np
import open3d as o3d
from typing import Tuple


def toTensor(array: np.ndarray, device: str = 'cpu', dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(array)).to(device, dtype=dtype)

def toNumpy(tensor: torch.Tensor, dtype = np.float32) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype)

def toO3DMesh(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    return o3d_mesh

def toNormalizeTransform(
    points: np.ndarray
) -> Tuple[np.ndarray, float]:
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    center = (min_bound + max_bound) / 2.0
    length = np.max(max_bound - min_bound)
    scale = 0.9 / length
    return center, scale

def transMesh(
    mesh: o3d.geometry.TriangleMesh,
    center: np.ndarray,
    scale: float,
    is_inverse: bool = False,
) -> bool:
    vertices = np.asarray(mesh.vertices)

    if is_inverse:
        vertices = vertices / scale + center
    else:
        vertices = (vertices - center) * scale

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return True
