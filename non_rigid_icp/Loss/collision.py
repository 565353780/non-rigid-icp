"""Differentiable self-collision barrier loss.

Given a (small) candidate set of non-adjacent face pairs produced by the broad
phase, penalize pairs whose triangle-triangle separation drops below a safety
margin. Minimizing this keeps opposing sheets apart *before* they cross, which
is the soft counterpart of the hard rollback guard in the fitter.
"""

import torch
from typing import Union

from non_rigid_icp.Method.collision import gatherTrianglePairs
from non_rigid_icp.Method.geometry import triangleTriangleDistance2


def selfCollisionBarrierLoss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    candidate_pairs: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Quadratic hinge barrier on candidate face-pair separation.

    loss = mean( relu(margin - dist)^2 ) over candidate pairs, where dist is the
    triangle-triangle separation. Differentiable w.r.t. `vertices` (hence the
    per-vertex displacement). Returns 0 when there are no candidates.

    Args:
        vertices: (V, 3) float tensor (deformed positions, requires grad ok).
        faces: (F, 3) long tensor.
        candidate_pairs: (P, 2) long, non-adjacent near pairs.
        margin: safety distance below which the barrier activates.
    """
    if candidate_pairs is None or candidate_pairs.shape[0] == 0:
        return vertices.sum() * 0.0

    tri1, tri2 = gatherTrianglePairs(vertices, faces, candidate_pairs)
    dist2 = triangleTriangleDistance2(tri1, tri2)
    dist = torch.sqrt(dist2 + 1e-18)
    viol = torch.clamp(margin - dist, min=0.0)
    # Normalize by the violating count (not the full candidate set) so the
    # per-pair barrier force does not vanish when most candidates are still safe.
    n_active = (viol > 0).sum().clamp(min=1).to(viol.dtype)
    return (viol ** 2).sum() / n_active
