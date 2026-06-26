"""Implicit field over the target mesh for closest-point (MLS-style) projection.

This is the projection backend for `Module/projection_fitter.py`: no optimization,
no IPC, no ARAP. It wraps Open3D's `RaycastingScene` (Embree, exact) and exposes:

  * the closest surface point of a query (the projection target), and
  * the winding-number signed distance phi (inside/outside sign).

Together these give the projection operator

    v' = v - phi(v) * grad phi(v).

For an exact triangle-mesh SDF the closest point IS ``v - phi*grad(phi)`` (one
Newton step lands exactly on the zero level set), so `closestPoints` is the
analytic result of that update and `project` simply caps the displacement to a
maximum step so a vertex can never jump across to a wrong layer.

All queries are streamed in bounded chunks so tens of millions of vertices fit
in memory. The scene itself lives on the CPU (Embree); only the small per-chunk
transfers cross the device boundary.
"""

import numpy as np
import open3d as o3d
import torch
from typing import Tuple, Union

EPS = 1e-20


def _to_numpy(x: Union[torch.Tensor, np.ndarray], dtype) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.ascontiguousarray(arr.astype(dtype))


def clampNorm(
    vectors: torch.Tensor, max_norm: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Clamp each row of `vectors` to length <= max_norm (per-row or scalar)."""
    norm = vectors.norm(dim=-1, keepdim=True)
    if isinstance(max_norm, torch.Tensor):
        cap = max_norm.reshape(-1, 1)
    else:
        cap = torch.as_tensor(float(max_norm), device=vectors.device).reshape(1, 1)
    scale = torch.clamp(cap / (norm + EPS), max=1.0)
    return vectors * scale


class ImplicitField(object):
    """Closest-point + signed-distance field of a target triangle mesh."""

    def __init__(
        self,
        vertices: Union[torch.Tensor, np.ndarray],
        faces: Union[torch.Tensor, np.ndarray],
        device: str = "cuda",
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        v = _to_numpy(vertices, np.float32)
        f = _to_numpy(faces, np.uint32)
        self._scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(v)
        mesh.triangle.indices = o3d.core.Tensor(f)
        self._geom_id = self._scene.add_triangles(mesh)

    def _as_query(self, query: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return _to_numpy(query, np.float32).reshape(-1, 3)

    def closestPoints(
        self, query: Union[torch.Tensor, np.ndarray], chunk: int = 2_000_000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closest surface point of each query.

        Returns:
            points: (N, 3) float32 on `self.device`.
            prim_ids: (N,) long, hit triangle id.
            normals: (N, 3) float32, the hit triangle's normal.
        """
        q = self._as_query(query)
        n = q.shape[0]
        pts = np.empty((n, 3), dtype=np.float32)
        prim = np.empty((n,), dtype=np.int64)
        nrm = np.empty((n, 3), dtype=np.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            ans = self._scene.compute_closest_points(o3d.core.Tensor(q[start:end]))
            pts[start:end] = ans["points"].numpy()
            prim[start:end] = ans["primitive_ids"].numpy().astype(np.int64)
            nrm[start:end] = ans["primitive_normals"].numpy()
        dev = self.device
        return (
            torch.from_numpy(pts).to(dev),
            torch.from_numpy(prim).to(dev),
            torch.from_numpy(nrm).to(dev),
        )

    def signedDistance(
        self, query: Union[torch.Tensor, np.ndarray], chunk: int = 2_000_000
    ) -> torch.Tensor:
        """Winding-number signed distance phi (negative inside). (N,) float32."""
        q = self._as_query(query)
        n = q.shape[0]
        out = np.empty((n,), dtype=np.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sd = self._scene.compute_signed_distance(o3d.core.Tensor(q[start:end]))
            out[start:end] = sd.numpy()
        return torch.from_numpy(out).to(self.device)

    def project(
        self,
        query: torch.Tensor,
        max_step: Union[torch.Tensor, float],
        chunk: int = 2_000_000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step each query toward its closest surface point, capped by max_step.

        Returns (projected, full_step_norm) where `full_step_norm` (N,) is the
        UNCAPPED distance to the surface (== |phi|), useful for convergence /
        error tracking. `max_step` may be a scalar or a per-vertex (N,) tensor.
        """
        cp, _, _ = self.closestPoints(query, chunk=chunk)
        step = cp - query
        full = step.norm(dim=-1)
        return query + clampNorm(step, max_step), full
