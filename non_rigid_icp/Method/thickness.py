"""Per-face / per-vertex wall thickness via normal ray casting.

A thin double-layer shell is connected through the surface normal: shooting a ray
from a face centroid along +/- normal hits the opposite layer at a distance equal
to the local wall thickness. This is O(F), exact, and INDEPENDENT of thickness
(unlike a proximity broad phase whose cost explodes with the search radius). The
projection fitter uses this to (a) flag thin-shell vertices needing extra care
and (b) cap each vertex's step to a fraction of its local thickness so the two
layers can move toward the target without ever closing the gap to zero.

Mirrors the ray-casting in `Method/sheet_constraints.detectWallPairsRaycast` but
returns the thickness scalar instead of the partner pairs.
"""

import numpy as np
import open3d as o3d
import torch
from typing import Tuple, Union

from non_rigid_icp.Method.geometry import vertexNormals


def faceWallThickness(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_thickness: float,
    eps: Union[float, None] = None,
) -> torch.Tensor:
    """Per-face local wall thickness (distance to the opposite layer).

    Returns (F,) float; entries with no opposite layer within `max_thickness`
    are set to +inf (treated as "not thin").
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
    nrm = normals.cpu().numpy().astype(np.float32)
    f = faces.shape[0]
    self_ids = np.arange(f, dtype=np.int64)
    INVALID = np.iinfo(np.uint32).max

    best_t = np.full(f, np.inf, dtype=np.float32)
    for sign in (-1.0, 1.0):
        origins = c + sign * eps * nrm
        dirs = (sign * nrm).astype(np.float32)
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
    return torch.from_numpy(best_t).to(device)


def faceWallPartnerDir(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_thickness: float,
    eps: Union[float, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-face wall thickness AND unit direction toward the partner layer.

    Same normal ray cast as `faceWallThickness`, but also returns the signed
    normal (``sign * face_normal``) of the ray that found the partner -- i.e. the
    direction from this face toward its opposite layer. This is what the
    projection fitter caps motion along (winding-robust: it is the ray that
    actually hit, regardless of how the mesh is wound). No-partner faces get
    thickness +inf and direction 0.
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
    nrm = normals.cpu().numpy().astype(np.float32)
    f = faces.shape[0]
    self_ids = np.arange(f, dtype=np.int64)
    INVALID = np.iinfo(np.uint32).max

    best_t = np.full(f, np.inf, dtype=np.float32)
    best_sign = np.zeros(f, dtype=np.float32)
    for sign in (-1.0, 1.0):
        origins = c + sign * eps * nrm
        dirs = (sign * nrm).astype(np.float32)
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
        best_sign[ok] = sign
    thickness = torch.from_numpy(best_t).to(device)
    direction = torch.from_numpy(best_sign).to(device).unsqueeze(1) * normals
    return thickness, direction


def vertexWallThickness(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_thickness: float,
    eps: Union[float, None] = None,
) -> torch.Tensor:
    """Per-vertex thickness = min thickness over incident faces (+inf if none).

    The min is conservative: a vertex is treated as thin if ANY of its faces is
    thin, so its step is capped tightly even at a thin/thick boundary.
    """
    face_t = faceWallThickness(vertices, faces, max_thickness, eps=eps)
    v = vertices.shape[0]
    out = torch.full((v,), float("inf"), device=vertices.device)
    out.scatter_reduce_(
        0, faces.reshape(-1), face_t.repeat_interleave(3), reduce="amin",
        include_self=True,
    )
    return out


def vertexWallPartner(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_thickness: float,
    eps: Union[float, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-vertex wall thickness (min over incident faces) and unit toward-partner
    direction (mean of incident faces' partner directions). No-partner vertices
    get thickness +inf and direction 0."""
    face_t, face_dir = faceWallPartnerDir(vertices, faces, max_thickness, eps=eps)
    v = vertices.shape[0]
    thickness = torch.full((v,), float("inf"), device=vertices.device)
    thickness.scatter_reduce_(
        0, faces.reshape(-1), face_t.repeat_interleave(3), reduce="amin",
        include_self=True,
    )
    dir_sum = torch.zeros(v, 3, device=vertices.device)
    flat = faces.reshape(-1)
    dir_sum.index_add_(0, flat, face_dir.repeat_interleave(3, dim=0))
    direction = dir_sum / (dir_sum.norm(dim=1, keepdim=True) + 1e-20)
    return thickness, direction


def inwardComponentCap(
    proposed: torch.Tensor,
    ref: torch.Tensor,
    toward_dir: torch.Tensor,
    allowance: torch.Tensor,
) -> torch.Tensor:
    """Cap each vertex's CUMULATIVE motion toward its wall partner.

    First principles for thin-shell collapse: the gap closes only via the
    displacement component along ``toward_dir`` (toward the opposite layer).
    Bounding that component (cumulative, from rest) to ``allowance =
    (thickness - margin)/2`` guarantees that even if BOTH layers move their full
    allowance toward each other the residual gap is still ``margin`` -- while
    tangential slide and OUTWARD motion (translation away from the partner, the
    beneficial same-direction case) are left completely free. Vertices with no
    partner (toward_dir = 0) are unaffected.

    Returns corrected positions (a new tensor).
    """
    disp = proposed - ref
    toward = (disp * toward_dir).sum(dim=1)
    excess = torch.clamp(toward - allowance, min=0.0)
    return proposed - excess.unsqueeze(1) * toward_dir
