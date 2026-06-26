"""Detect thin / double-layer sheet pairs to constrain during fitting.

A "sheet pair" is a non-adjacent triangle pair that is currently close (within
`gap_max`) -- typically the two layers of the thin closed shell, or the two
sides of a fold. We freeze, per pair, the separation axis a = unit(c_j - c_i)
and feed (pairs, axis) to `Loss.sheet.sheetOrderBarrierLoss`, which keeps the
layers from crossing while letting them compress toward the target.

Built on the same complete inflated-AABB broad phase as the collision guard, so
it can never miss the opposing layer (the failure mode of centroid k-NN).
"""

import numpy as np
import open3d as o3d
import torch
from typing import Tuple, Union

from non_rigid_icp.Method.collision import buildCollisionCandidatesAABB
from non_rigid_icp.Method.topology import facePairsShareVertex


def faceCentroidsNormals(
    vertices: torch.Tensor, faces: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    tri = vertices[faces]
    centroids = tri.mean(dim=1)
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-20)
    return centroids, normals


def detectSheetPairs(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    gap_max: float,
    active_face_mask: Union[torch.Tensor, None] = None,
    opposite_only: bool = False,
    normal_dot_max: float = -0.3,
    max_pairs: Union[int, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find near non-adjacent face pairs and their frozen separation axes.

    Args:
        gap_max: AABB inflation = maximum gap to treat as a sheet pair.
        active_face_mask: optional (F,) bool; restrict to pairs touching it.
        opposite_only: if True keep only pairs with normals facing each other
            (dot < normal_dot_max) -- the genuine double-layer case. If False,
            keep all near non-adjacent pairs (also constrains folds).
        max_pairs: optional cap (closest pairs kept).

    Returns:
        pairs: (P, 2) long.
        axis: (P, 3) float, unit(c_j - c_i), detached.
    """
    pairs = buildCollisionCandidatesAABB(
        vertices,
        faces,
        margin=gap_max,
        active_face_mask=active_face_mask,
        max_pairs=max_pairs,
    )
    if pairs.shape[0] == 0:
        return pairs, torch.zeros(0, 3, device=vertices.device)

    centroids, normals = faceCentroidsNormals(vertices.detach(), faces)
    if opposite_only:
        dot = (normals[pairs[:, 0]] * normals[pairs[:, 1]]).sum(dim=1)
        keep = dot < normal_dot_max
        pairs = pairs[keep]
        if pairs.shape[0] == 0:
            return pairs, torch.zeros(0, 3, device=vertices.device)

    diff = centroids[pairs[:, 1]] - centroids[pairs[:, 0]]
    axis = diff / (diff.norm(dim=1, keepdim=True) + 1e-20)
    return pairs, axis.detach()


def detectWallPairsRaycast(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_thickness: float,
    eps: Union[float, None] = None,
) -> torch.Tensor:
    """Find each face's opposite-layer WALL PARTNER by normal ray casting.

    A thin double layer is connected through the surface normal: shooting a ray
    from a face centroid along -normal (into the wall) hits the partner face on
    the opposite layer at a distance equal to the local wall thickness. This is
    O(F), EXACT, and -- crucially -- INDEPENDENT of wall thickness, unlike a
    proximity broad phase whose cost explodes with the search radius and which
    silently misses walls thicker than its gap (the failure that let case1
    collapse). Rays are cast both inward and outward; the nearer valid hit within
    `max_thickness` is kept.

    Args:
        vertices: (V, 3) -- use the CLEAN reference so the layers are still
            correctly ordered (the returned pairs carry no axis; the caller
            freezes it from this same clean geometry).
        faces: (F, 3) long, consistent winding.
        max_thickness: ignore hits farther than this (skips rays that shoot
            across a cavity and hit an unrelated far wall).
        eps: ray-origin offset off the surface to avoid self-hits; defaults to a
            small fraction of the median triangle size.

    Returns:
        (P, 2) long sorted unique non-adjacent wall pairs on `faces.device`.
    """
    device = faces.device
    tri = vertices.detach()[faces]
    centroids = tri.mean(dim=1)
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-20)
    if eps is None:
        edge = (tri[:, 1] - tri[:, 0]).norm(dim=1)
        eps = 0.05 * float(edge.median().clamp(min=1e-9))

    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(
        vertices.detach().cpu().numpy().astype(np.float32)
    )
    mesh.triangle.indices = o3d.core.Tensor(
        faces.detach().cpu().numpy().astype(np.uint32)
    )
    scene.add_triangles(mesh)

    c = centroids.cpu().numpy().astype(np.float32)
    n = normals.cpu().numpy().astype(np.float32)
    F = faces.shape[0]
    self_ids = np.arange(F, dtype=np.int64)
    INVALID = np.iinfo(np.uint32).max

    best_t = np.full(F, np.inf, dtype=np.float32)
    best_j = np.full(F, -1, dtype=np.int64)
    for sign in (-1.0, 1.0):
        origins = c + sign * eps * n
        dirs = (sign * n).astype(np.float32)
        rays = o3d.core.Tensor(
            np.concatenate([origins, dirs], axis=1).astype(np.float32)
        )
        ans = scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        ok = (
            np.isfinite(t_hit)
            & (prim != INVALID)
            & (prim != self_ids)
            & (t_hit < max_thickness)
            & (t_hit < best_t)
        )
        best_t[ok] = t_hit[ok]
        best_j[ok] = prim[ok]

    has = best_j >= 0
    i = self_ids[has]
    j = best_j[has]
    pairs = np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1)
    pairs_t = torch.from_numpy(pairs).to(device)
    pairs_t = torch.unique(pairs_t, dim=0)
    if pairs_t.shape[0] == 0:
        return pairs_t
    shared = facePairsShareVertex(pairs_t, faces)
    return pairs_t[~shared]
