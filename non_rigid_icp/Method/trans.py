import torch
import trimesh
import numpy as np
import open3d as o3d
from typing import Tuple, Union

def toTensor(array: Union[np.ndarray, torch.Tensor], device: str = 'cpu', dtype = torch.float32) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.tensor(np.asarray(array))
    return tensor.to(device, dtype=dtype)

def toNumpy(tensor: Union[torch.Tensor, np.ndarray], dtype = np.float32) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    return array.astype(dtype)

def toPointCloud(points: Union[torch.Tensor, np.ndarray]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(toNumpy(points))
    return pcd

def toMesh(
    vertices: Union[torch.Tensor, np.ndarray],
    triangles: Union[torch.Tensor, np.ndarray],
) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(toNumpy(vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(toNumpy(triangles, np.int64))
    return o3d_mesh

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

def transPoints(
    points: np.ndarray,
    center: np.ndarray,
    scale: float,
    is_inverse: bool = False,
) -> np.ndarray:
    if is_inverse:
        return points / scale + center

    return (points - center) * scale

def transPcd(
    pcd: o3d.geometry.PointCloud,
    center: np.ndarray,
    scale: float,
    is_inverse: bool = False,
) -> bool:
    points = np.asarray(pcd.points)

    trans_points = transPoints(points, center, scale, is_inverse)

    pcd.points = o3d.utility.Vector3dVector(trans_points)
    return True

def transMesh(
    mesh: o3d.geometry.TriangleMesh,
    center: np.ndarray,
    scale: float,
    is_inverse: bool = False,
) -> bool:
    vertices = np.asarray(mesh.vertices)

    trans_vertices = transPoints(vertices, center, scale, is_inverse)

    mesh.vertices = o3d.utility.Vector3dVector(trans_vertices)
    return True

def transGeometry(
    geometry: Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud],
    center: np.ndarray,
    scale: float,
    is_inverse: bool = False,
) -> bool:
    if isinstance(geometry, o3d.geometry.PointCloud):
        return transPcd(geometry, center, scale, is_inverse)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        return transMesh(geometry, center, scale, is_inverse)
    else:
        return geometry.transform(center, scale, is_inverse)
