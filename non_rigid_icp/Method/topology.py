"""Reusable mesh-topology atomic functions.

These operate on plain tensors (vertices/faces) so they can be shared across the
fitter, the self-collision guard, the error-region localizer and the local
subdivision. Everything is vectorized (no python per-element loops) so it scales
to the tens-of-millions-of-faces meshes this project targets.
"""

import torch
import numpy as np
from typing import Tuple, Union

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _scipy_cc
except Exception as _e:  # pragma: no cover - scipy is expected in the env
    coo_matrix = None
    _scipy_cc = None


def buildUniqueEdges(triangles: torch.Tensor) -> torch.Tensor:
    """Unique undirected edges of a triangle mesh.

    Args:
        triangles: (F, 3) long tensor.

    Returns:
        edges: (E, 2) long tensor, each row sorted (e[0] < e[1]), de-duplicated.
    """
    e = torch.cat(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]], dim=0
    )
    e = torch.sort(e, dim=1).values
    edges = torch.unique(e, dim=0)
    return edges


def buildVertexToFaces(
    triangles: torch.Tensor, num_vertices: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CSR-style vertex -> incident faces map.

    Returns:
        offsets: (num_vertices + 1,) long. Faces of vertex v are
            indices[offsets[v]:offsets[v + 1]].
        indices: (3F,) long, face ids grouped by vertex.
    """
    f = triangles.shape[0]
    v = triangles.reshape(-1)
    fid = torch.arange(f, device=triangles.device).repeat_interleave(3)
    order = torch.argsort(v)
    indices = fid[order]
    counts = torch.bincount(v, minlength=num_vertices)
    offsets = torch.zeros(num_vertices + 1, dtype=torch.long, device=triangles.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets, indices


def buildFaceAdjacency(triangles: torch.Tensor) -> torch.Tensor:
    """Edge-adjacency between faces (faces sharing an undirected edge).

    Returns:
        pairs: (A, 2) long tensor of adjacent face id pairs. For a manifold mesh
        each interior edge contributes exactly one pair. Non-manifold edges
        (>2 faces) are linked consecutively, which is sufficient for dilation
        and connected-component grouping.
    """
    f = triangles.shape[0]
    # per-face edge stack so face ids stay aligned with edges (face-major order)
    per_face = torch.stack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        dim=1,
    )  # (F, 3, 2)
    per_face = torch.sort(per_face, dim=2).values
    e = per_face.reshape(-1, 2)
    fid = torch.arange(f, device=triangles.device).repeat_interleave(3)
    # encode the undirected edge as a single key for sorting
    key = e[:, 0].to(torch.int64) * (int(triangles.max().item()) + 1) + e[:, 1].to(
        torch.int64
    )
    order = torch.argsort(key)
    key_sorted = key[order]
    fid_sorted = fid[order]
    same = key_sorted[1:] == key_sorted[:-1]
    a = fid_sorted[:-1][same]
    b = fid_sorted[1:][same]
    pairs = torch.stack([a, b], dim=1)
    return pairs


def dilateFaceMask(
    face_mask: torch.Tensor, face_adjacency: torch.Tensor, n_rings: int = 1
) -> torch.Tensor:
    """Grow a boolean face mask by n_rings along the face adjacency graph."""
    if n_rings <= 0 or face_adjacency.numel() == 0:
        return face_mask
    mask = face_mask.clone()
    a = face_adjacency[:, 0]
    b = face_adjacency[:, 1]
    for _ in range(n_rings):
        grown = mask.clone()
        grown[a] |= mask[b]
        grown[b] |= mask[a]
        mask = grown
    return mask


def facePairsShareVertex(
    face_pairs: torch.Tensor, triangles: torch.Tensor
) -> torch.Tensor:
    """Boolean mask: which (fi, fj) pairs share at least one vertex.

    Args:
        face_pairs: (P, 2) long.
        triangles: (F, 3) long.

    Returns:
        shared: (P,) bool, True if the two faces are topologically adjacent
        (share >= 1 vertex), i.e. they must be excluded from self-collision
        candidates.
    """
    if face_pairs.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=face_pairs.device)
    fi = triangles[face_pairs[:, 0]]  # (P, 3)
    fj = triangles[face_pairs[:, 1]]  # (P, 3)
    # (P, 3, 3) all-vs-all vertex comparison
    shared = (fi.unsqueeze(2) == fj.unsqueeze(1)).any(dim=2).any(dim=1)
    return shared


def connectedFaceComponents(
    face_mask: Union[torch.Tensor, np.ndarray],
    face_adjacency: Union[torch.Tensor, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Connected components of the masked sub-graph of faces.

    Args:
        face_mask: (F,) bool.
        face_adjacency: (A, 2) edge-adjacency pairs.

    Returns:
        labels: (F,) int32, component id per masked face, -1 for unmasked faces.
        sizes: (num_components,) int32, number of faces in each component.
    """
    if isinstance(face_mask, torch.Tensor):
        mask_np = face_mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = np.asarray(face_mask, dtype=bool)
    if isinstance(face_adjacency, torch.Tensor):
        adj_np = face_adjacency.detach().cpu().numpy().astype(np.int64)
    else:
        adj_np = np.asarray(face_adjacency, dtype=np.int64)

    f = mask_np.shape[0]
    labels = np.full(f, -1, dtype=np.int32)
    masked_ids = np.nonzero(mask_np)[0]
    if masked_ids.size == 0:
        return labels, np.zeros(0, dtype=np.int32)

    # keep only adjacency edges with both endpoints masked, then relabel locally
    keep = mask_np[adj_np[:, 0]] & mask_np[adj_np[:, 1]]
    sub = adj_np[keep]

    remap = np.full(f, -1, dtype=np.int64)
    remap[masked_ids] = np.arange(masked_ids.size)
    n = masked_ids.size

    if _scipy_cc is None or coo_matrix is None:
        # fallback: every masked face its own component
        labels[masked_ids] = np.arange(n, dtype=np.int32)
        return labels, np.ones(n, dtype=np.int32)

    if sub.size == 0:
        local = np.arange(n, dtype=np.int64)
        n_comp = n
    else:
        rows = remap[sub[:, 0]]
        cols = remap[sub[:, 1]]
        data = np.ones(rows.size, dtype=np.int8)
        graph = coo_matrix((data, (rows, cols)), shape=(n, n))
        n_comp, local = _scipy_cc(graph, directed=False)

    labels[masked_ids] = local.astype(np.int32)
    sizes = np.bincount(local, minlength=n_comp).astype(np.int32)
    return labels, sizes
