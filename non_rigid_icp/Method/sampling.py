import torch
import numpy as np
import open3d as o3d
from typing import Tuple, Union

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.trans import toTensor, toNumpy


def toTargetFrameTransform(target_vertices: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Compute a normalization transform anchored on the target mesh bbox.

    Both source and target are normalized with the SAME center/scale so that
    the optimization happens in a shared frame and the evaluation threshold
    tau = L / 2048 (with L the largest bbox edge of the ORIGINAL target) can be
    expressed consistently in the normalized frame as tau * scale.

    Returns:
        center: (3,) bbox center of the target (original coords)
        scale: float, normalization scale (normalized = (v - center) * scale)
        L: float, largest bbox edge length of the original target
    """
    min_bound = np.min(target_vertices, axis=0)
    max_bound = np.max(target_vertices, axis=0)
    center = (min_bound + max_bound) / 2.0
    extent = max_bound - min_bound
    L = float(np.max(extent))
    # keep the largest edge at 0.9 in normalized space, matching Mesh.normalize convention
    scale = 0.9 / L if L > 1e-12 else 1.0
    return center.astype(np.float64), float(scale), L


def normalizePairToTargetFrame(
    source_mesh: Mesh, target_mesh: Mesh
) -> Tuple[np.ndarray, float, float]:
    """Normalize source and target meshes in-place into the target bbox frame.

    Returns (center, scale, L) so the deformed result can be transformed back to
    the original target coordinates via Mesh.transform(center, scale, is_inverse=True).
    """
    center, scale, L = toTargetFrameTransform(target_mesh.vertices)

    source_mesh.transform(center, scale, is_inverse=False)
    source_mesh.norm_center = center
    source_mesh.norm_scale = scale
    source_mesh.is_normalized = True

    target_mesh.transform(center, scale, is_inverse=False)
    target_mesh.norm_center = center
    target_mesh.norm_scale = scale
    target_mesh.is_normalized = True

    return center, scale, L


def sampleMeshSurface(
    mesh: Union[Mesh, o3d.geometry.TriangleMesh],
    n_points: int,
    method: str = "uniform",
    seed: Union[int, None] = None,
    with_normals: bool = False,
):
    """Uniformly sample points on a mesh surface.

    Returns points (N,3) float32; if with_normals also returns normals (N,3).
    """
    if isinstance(mesh, Mesh):
        o3d_mesh = mesh.toO3DMesh()
    else:
        o3d_mesh = mesh

    if with_normals and not o3d_mesh.has_vertex_normals():
        o3d_mesh.compute_vertex_normals()

    if seed is not None:
        o3d.utility.random.seed(seed)

    if method == "poisson":
        pcd = o3d_mesh.sample_points_poisson_disk(n_points)
    else:
        pcd = o3d_mesh.sample_points_uniformly(n_points)

    points = np.asarray(pcd.points, dtype=np.float32)
    if with_normals:
        normals = np.asarray(pcd.normals, dtype=np.float32)
        return points, normals
    return points


def buildBarycentricSamples(
    triangles: np.ndarray,
    n_points: int,
    vertices: Union[np.ndarray, None] = None,
    seed: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-sample fixed face ids + barycentric weights on a source mesh.

    These can be reused every iteration to turn deformed vertices into a
    differentiable surface point cloud via sampleDeformedSurface.

    If vertices is provided, faces are sampled proportionally to their area so
    the resulting point cloud is (approximately) area-uniform.

    Returns:
        face_ids: (n_points,) int64
        barys: (n_points, 3) float32, rows sum to 1
    """
    rng = np.random.default_rng(seed)

    n_faces = triangles.shape[0]
    if vertices is not None:
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total = areas.sum()
        if total <= 1e-20:
            probs = None
        else:
            probs = areas / total
    else:
        probs = None

    face_ids = rng.choice(n_faces, size=n_points, p=probs).astype(np.int64)

    r1 = rng.random(n_points, dtype=np.float64)
    r2 = rng.random(n_points, dtype=np.float64)
    sqrt_r1 = np.sqrt(r1)
    b0 = 1.0 - sqrt_r1
    b1 = sqrt_r1 * (1.0 - r2)
    b2 = sqrt_r1 * r2
    barys = np.stack([b0, b1, b2], axis=1).astype(np.float32)
    return face_ids, barys


def sampleDeformedSurface(
    deformed_vertices: torch.Tensor,
    triangles: torch.Tensor,
    face_ids: torch.Tensor,
    barys: torch.Tensor,
) -> torch.Tensor:
    """Differentiably build a surface point cloud from deformed vertices.

    Args:
        deformed_vertices: (V, 3) tensor (requires grad ok)
        triangles: (F, 3) int64 tensor
        face_ids: (N,) int64 tensor of sampled faces
        barys: (N, 3) float32 tensor of barycentric weights

    Returns:
        points: (N, 3) tensor on the deformed surface
    """
    tri = triangles[face_ids]  # (N, 3)
    v0 = deformed_vertices[tri[:, 0]]
    v1 = deformed_vertices[tri[:, 1]]
    v2 = deformed_vertices[tri[:, 2]]
    points = (
        barys[:, 0:1] * v0 + barys[:, 1:2] * v1 + barys[:, 2:3] * v2
    )
    return points
