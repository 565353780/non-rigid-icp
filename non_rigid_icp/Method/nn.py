import numpy as np
import open3d as o3d
import open3d.core as o3c
from typing import Tuple, Union

import torch


def _toO3CTensor(points, device_str: str) -> o3c.Tensor:
    if isinstance(points, torch.Tensor):
        arr = points.detach().cpu().numpy()
    else:
        arr = np.asarray(points)
    arr = np.ascontiguousarray(arr.astype(np.float32))
    if device_str.startswith("cuda"):
        try:
            dev = o3c.Device("CUDA:0")
            return o3c.Tensor(arr, device=dev)
        except Exception:
            pass
    return o3c.Tensor(arr, device=o3c.Device("CPU:0"))


class NNIndex(object):
    """Spatial nearest-neighbor index over a reference point set.

    Uses Open3D's NearestNeighborSearch (GPU when available), giving O(log N)
    queries instead of the O(N*M) brute-force chamfer op. This is what makes
    full-resolution (tens of millions of vertices) correspondence tractable.
    """

    def __init__(self, reference_points, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self._ref_t = _toO3CTensor(reference_points, self.device)
        self._nns = o3d.core.nns.NearestNeighborSearch(self._ref_t)
        self._nns.knn_index()

    def query(self, query_points, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices (N,), squared_distances (N,)) for k=1."""
        q = _toO3CTensor(query_points, self.device)
        idx, dist2 = self._nns.knn_search(q, k)
        idx_np = idx.cpu().numpy().reshape(-1).astype(np.int64)
        dist2_np = dist2.cpu().numpy().reshape(-1).astype(np.float32)
        return idx_np, dist2_np

    def queryKNN(self, query_points, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices (N, k), squared_distances (N, k)) for arbitrary k."""
        q = _toO3CTensor(query_points, self.device)
        idx, dist2 = self._nns.knn_search(q, k)
        idx_np = idx.cpu().numpy().reshape(-1, k).astype(np.int64)
        dist2_np = dist2.cpu().numpy().reshape(-1, k).astype(np.float32)
        return idx_np, dist2_np


def nearestIndices(
    query_points,
    reference_points,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience one-shot nearest-neighbor: returns (indices, squared_dists)."""
    index = NNIndex(reference_points, device=device)
    return index.query(query_points, k=1)
