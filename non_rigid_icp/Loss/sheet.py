"""Sheet order / separation barrier.

The watertight source is a thin closed shell: most surfaces come as two
opposing layers separated by ~1-4 tau. To reach a high F1 both layers must sit
within tau of the single target surface, i.e. they must compress to within ~2
tau of each other -- but they must NOT pass *through* each other (which is what
produced the 64.8%-self-intersecting result).

Distance-only barriers are symmetric and tunneling-prone: once two sheets touch,
a single optimization step can push them to the wrong side. This barrier instead
freezes, at detection time, the *separation axis* a = unit(c_j - c_i) between a
near pair of face centroids, and penalizes the signed separation s = (c_j - c_i)
. a dropping below a small `min_margin`. Because s is monotonic in the relative
position along a, the layers can compress down to `min_margin` (good for F1) but
the barrier grows without bound as they try to cross (s -> 0 and beyond), which
robustly preserves the original layer ordering.
"""

import torch


def sheetOrderBarrierLoss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pairs: torch.Tensor,
    axis: torch.Tensor,
    min_margin: float,
) -> torch.Tensor:
    """Quadratic hinge on the signed separation of near face-centroid pairs.

    Args:
        vertices: (V, 3) deformed positions (grad ok).
        faces: (F, 3) long.
        pairs: (P, 2) long, non-adjacent near pairs (i, j).
        axis: (P, 3) float, frozen unit separation axis = unit(c_j - c_i) at
            detection time (detached, constant).
        min_margin: minimum allowed signed separation; below it the barrier
            activates. Use a small fraction of tau so layers may nearly touch
            without crossing.

    Returns:
        scalar loss = mean( relu(min_margin - s)^2 ), s = (c_j - c_i) . axis.
    """
    if pairs is None or pairs.shape[0] == 0:
        return vertices.sum() * 0.0
    tri = vertices[faces]  # (F, 3, 3)
    centroids = tri.mean(dim=1)  # (F, 3)
    ci = centroids[pairs[:, 0]]
    cj = centroids[pairs[:, 1]]
    s = ((cj - ci) * axis).sum(dim=1)
    viol = torch.clamp(min_margin - s, min=0.0)
    # Normalize by the number of *violating* pairs, not the (huge) total. A plain
    # mean over millions of mostly-satisfied pairs drives the per-violation
    # gradient to ~0, so the barrier silently fails on a densely double-layered
    # mesh. Dividing by the active count keeps each crossing pair at O(1) force.
    n_active = (viol > 0).sum().clamp(min=1).to(viol.dtype)
    return (viol ** 2).sum() / n_active
