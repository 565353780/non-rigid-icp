"""Scalable stiffness (shape-preservation) step for the per-step projection fit.

`Model/deform.py` realizes the Amberg et al. stiffness constraint with a
per-vertex affine field (R_i, t_i) and the regularizer

    E_stiff = sum_{(i,j) in edges} ||R_i - R_j||^2 + ||t_i - t_j||^2 ,

i.e. ADJACENT VERTICES MUST UNDERGO CONSISTENT LOCAL TRANSFORMS so the surface
deforms smoothly and locally rigidly (no shearing/spiking). That class stores a
dense 3x3 matrix per vertex and builds its edge set with a python loop, which is
fine for the small clinical meshes it targets but does not scale to the case1
mesh (14M+ vertices, 43M+ edges).

This module reuses the SAME PRINCIPLE through cheap vectorized atoms so it works
at that scale: for each vertex we fit the local rigid rotation R_i that best maps
its rest 1-ring to its current 1-ring (Procrustes / Kabsch on the covariance),
then take one damped Jacobi step of the ARAP/stiffness energy

    sum_{(i,j)} || (x_i - x_j) - 0.5 (R_i + R_j) (X_i - X_j) ||^2

over the CURRENT positions x (X = rest = `ref`). The fixed point of this
iteration is the as-rigid-as-possible surface, exactly the shape the Amberg
stiffness term rewards, but each step is O(V + E) tensor ops with no per-vertex
matrix parameters and no python loops.

Atoms (all batched, no python loops):
  * `fitLocalRotations` : per-vertex Kabsch rotation rest-1-ring -> current-1-ring
  * `stiffnessStep`     : one damped Jacobi ARAP/stiffness relaxation of positions
"""

import torch
from typing import Tuple

EPS = 1e-12


def _neighbor_sums(
    values: torch.Tensor, e0: torch.Tensor, e1: torch.Tensor, n: int
) -> torch.Tensor:
    """Sum `values[j]` over the neighbors j of each vertex i (undirected edges)."""
    out = torch.zeros((n,) + values.shape[1:], device=values.device, dtype=values.dtype)
    out.index_add_(0, e0, values[e1])
    out.index_add_(0, e1, values[e0])
    return out


def vertexDegrees(
    e0: torch.Tensor, e1: torch.Tensor, n: int, device, dtype=torch.float32
) -> torch.Tensor:
    deg = torch.zeros(n, device=device, dtype=dtype)
    ones = torch.ones(e0.shape[0], device=device, dtype=dtype)
    deg.index_add_(0, e0, ones)
    deg.index_add_(0, e1, ones)
    return deg


def fitLocalRotations(
    rest: torch.Tensor,
    cur: torch.Tensor,
    e0: torch.Tensor,
    e1: torch.Tensor,
) -> torch.Tensor:
    """Per-vertex rotation R_i (V,3,3) best mapping the REST 1-ring edge vectors
    to the CURRENT 1-ring edge vectors (ARAP local step, Kabsch via SVD).

    The covariance S_i = sum_{j~i} (X_j - X_i)(x_j - x_i)^T is accumulated with
    scatter-adds, then a single batched SVD gives R_i = V U^T (det-corrected).
    """
    n = rest.shape[0]
    # rest / current edge vectors (i -> j) for both edge orientations
    de_rest = rest[e1] - rest[e0]
    de_cur = cur[e1] - cur[e0]

    # S_i += sum over incident edges of (rest_edge)(cur_edge)^T, accumulated at
    # BOTH endpoints (sign cancels: (-a)(-b)^T = a b^T), giving the symmetric
    # 1-ring covariance per vertex.
    outer = de_rest.unsqueeze(2) * de_cur.unsqueeze(1)  # (E,3,3)
    S = torch.zeros((n, 3, 3), device=rest.device, dtype=rest.dtype)
    S.index_add_(0, e0, outer)
    S.index_add_(0, e1, outer)

    # Kabsch: R = V U^T with a sign fix so det(R) = +1 (no reflections).
    U, _, Vh = torch.linalg.svd(S + EPS * torch.eye(3, device=S.device))
    V = Vh.transpose(-1, -2)
    det = torch.linalg.det(torch.matmul(V, U.transpose(-1, -2)))
    D = torch.eye(3, device=S.device).repeat(n, 1, 1)
    D[:, 2, 2] = torch.sign(det)
    R = torch.matmul(torch.matmul(V, D), U.transpose(-1, -2))
    return R


def stiffnessStep(
    cur: torch.Tensor,
    rest: torch.Tensor,
    e0: torch.Tensor,
    e1: torch.Tensor,
    iters: int = 1,
    weight: float = 1.0,
    pinned_mask: torch.Tensor = None,
) -> torch.Tensor:
    """One (or `iters`) damped Jacobi relaxation(s) of the ARAP / Amberg stiffness
    energy, returning shape-regularized positions.

    For each edge the target relative position is the rest edge rotated by the
    average of the two endpoints' local rotations, so the update both SMOOTHS the
    displacement field and PRESERVES rest edge lengths (the geometric content of
    the R/t consistency term in `deform.py`). `weight` in [0,1] is the damping
    (0 = no change, 1 = full Jacobi step). Pinned vertices are held fixed.
    """
    if weight <= 0.0 or iters <= 0:
        return cur
    n = cur.shape[0]
    deg = vertexDegrees(e0, e1, n, cur.device, cur.dtype).clamp(min=1.0).unsqueeze(1)
    x = cur
    for _ in range(iters):
        R = fitLocalRotations(rest, x, e0, e1)
        de_rest = rest[e1] - rest[e0]  # (E,3) rest edge i->j
        # rotated rest edge using the per-edge averaged rotation
        Ravg = 0.5 * (R[e0] + R[e1])  # (E,3,3)
        b_ij = torch.matmul(Ravg, de_rest.unsqueeze(2)).squeeze(2)  # (E,3)
        # Jacobi target: x_i = mean_j ( x_j + R_avg (X_i - X_j) )
        #              = mean_j ( x_j - b_ij )   for the i->j orientation
        # accumulate neighbor x_j and the rotated-edge correction at both ends
        nb_x = _neighbor_sums(x, e0, e1, n)
        corr = torch.zeros_like(x)
        corr.index_add_(0, e0, -b_ij)  # i side: target uses (x_j - b_ij)
        corr.index_add_(0, e1, b_ij)   # j side: target uses (x_i + b_ij)
        target = (nb_x + corr) / deg
        x_new = (1.0 - weight) * x + weight * target
        if pinned_mask is not None:
            x_new[pinned_mask] = x[pinned_mask]
        x = x_new
    return x
