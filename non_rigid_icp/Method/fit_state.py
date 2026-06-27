"""Per-vertex optimization state for the clamped stepwise fit.

First principles: a vertex should keep contributing its data (fit) error to the
loss only while the optimizer can actually move it closer to the target. In a
watertight mesh with many thin / double-layer shells some vertices want to move
(their residual is large and the gradient asks for a step) but cannot: the
per-step clamp and the trajectory pull-back drag them straight back every step.
Those vertices are *unoptimizable at the current resolution* -- keeping their
unreachable distance in the loss only adds a constant pull that fights the rest
of the mesh and inflates the global residual without ever improving. We detect
them with a small batched state machine (no python per-vertex loops), drop them
from the data loss, and -- crucially -- exclude their stagnant-but-unreachable
regions from the adaptive subdivision (subdividing an unreachable thin shell
only multiplies faces without reducing error).

Everything here is plain-tensor in / plain-tensor out so it is reusable and unit
testable independent of the fitter, and it is carried through local subdivision
in lock-step with the geometry (new midpoints inherit a conservative blend of
their two parents' state) exactly like the clean reference field `_ref_verts`.

State tensors (all length V, on the mesh device):
  optimizable   (bool)  : include this vertex's distance error in the data loss.
  blocked_count (int32) : consecutive steps the vertex "wanted to move but did
                          not" (intended a real step yet ended up ~stationary).
  resid_prev    (float) : previous step residual d_i (tau-free, normalized frame).
  resid_drop_ema(float) : EMA of the per-step residual drop (resid_prev - resid),
                          the local "is the error still falling here?" signal.
  last_refine   (int32) : subdivision level at which the vertex was last created
                          / refined; used to cool down repeated refinement.
"""

import torch
from typing import Dict


def initVertexFitState(
    n_vertices: int, device: str, level: int = -1
) -> Dict[str, torch.Tensor]:
    """Fresh state for `n_vertices` vertices (all optimizable, no history).

    `last_refine` defaults to -1 ("never refined"), so the cool-down test
    `(level - last_refine) >= refine_cooldown` is satisfied at level 0 -- a
    pristine vertex is immediately eligible for its first refinement; only
    vertices touched by a refinement get their `last_refine` bumped (to the
    next level), which then blocks them for `refine_cooldown` rounds."""
    return {
        "optimizable": torch.ones(n_vertices, dtype=torch.bool, device=device),
        "blocked_count": torch.zeros(n_vertices, dtype=torch.int32, device=device),
        "resid_prev": torch.full((n_vertices,), float("inf"), device=device),
        "resid_drop_ema": torch.zeros(n_vertices, device=device),
        "last_refine": torch.full(
            (n_vertices,), int(level), dtype=torch.int32, device=device
        ),
    }


def updateVertexFitState(
    state: Dict[str, torch.Tensor],
    resid: torch.Tensor,
    intended_move: torch.Tensor,
    actual_move: torch.Tensor,
    tau: float,
    *,
    unopt_error_tau: float = 1.0,
    min_intended_move_tau: float = 0.02,
    min_actual_move_tau: float = 0.004,
    min_progress_ratio: float = 0.1,
    block_patience: int = 3,
    drop_ema_beta: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Advance the per-vertex state by one optimization step (fully batched).

    A vertex is "stalled this step" when it genuinely tried to move but ended up
    essentially in place:
        intended_move > min_intended_move_tau * tau   (the step wanted to move)
      AND its residual is still off the surface
        resid > unopt_error_tau * tau
      AND it barely moved
        actual_move < min_actual_move_tau * tau
        OR actual_move / intended_move < min_progress_ratio.

    `blocked_count` counts consecutive stalled steps and resets on any real
    progress. Once it reaches `block_patience` the vertex is marked
    NON-optimizable (removed from the data loss). The residual-drop EMA is
    maintained for every vertex as the local convergence signal the subdivision
    localizer consumes. Returns the same dict (mutated in place) for chaining.

    Args:
        resid:         (V,) current closest-point distance d_i (normalized frame).
        intended_move: (V,) pre-pullback desired step length (== clamped GD step).
        actual_move:   (V,) realized per-step coordinate change after pull-back.
        tau:           the L/2048 tolerance in the normalized frame.
    """
    prev = state["resid_prev"]
    finite_prev = torch.isfinite(prev)
    drop = torch.where(finite_prev, prev - resid, torch.zeros_like(resid))
    state["resid_drop_ema"] = torch.where(
        finite_prev,
        drop_ema_beta * state["resid_drop_ema"] + (1.0 - drop_ema_beta) * drop,
        state["resid_drop_ema"],
    )
    state["resid_prev"] = resid

    wants_move = intended_move > (min_intended_move_tau * tau)
    still_off = resid > (unopt_error_tau * tau)
    barely_moved = (actual_move < (min_actual_move_tau * tau)) | (
        actual_move < (min_progress_ratio * intended_move.clamp(min=1e-12))
    )
    stalled = wants_move & still_off & barely_moved

    inc = stalled.to(torch.int32)
    state["blocked_count"] = torch.where(
        stalled,
        state["blocked_count"] + inc,
        torch.zeros_like(state["blocked_count"]),
    )
    newly_blocked = state["blocked_count"] >= int(block_patience)
    state["optimizable"] = state["optimizable"] & (~newly_blocked)
    return state


def faceStateFromVertexState(
    state: Dict[str, torch.Tensor],
    faces: torch.Tensor,
    tau: float,
    *,
    local_drop_tau: float = 0.02,
    max_blocked_vertex_ratio: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Reduce the per-vertex state to per-face booleans for the refine localizer.

    Returns a dict of (F,) tensors:
        plateau_face : the face's residual is no longer dropping (mean drop EMA
                       over its 3 vertices < local_drop_tau * tau) -- the LOCAL
                       "stopped improving here" signal (vs the old global one).
        blocked_face : too large a fraction of the face's vertices are
                       non-optimizable -- an unreachable thin shell that
                       subdivision cannot help, so it must be skipped.
        last_refine  : max subdivision level over the face's vertices (cool-down).
    """
    drop = state["resid_drop_ema"][faces]           # (F, 3)
    plateau_face = drop.mean(dim=1) < (local_drop_tau * tau)

    blocked = (~state["optimizable"]).float()[faces]  # (F, 3)
    blocked_face = blocked.mean(dim=1) > max_blocked_vertex_ratio

    last_refine = state["last_refine"][faces].amax(dim=1)
    return {
        "plateau_face": plateau_face,
        "blocked_face": blocked_face,
        "last_refine": last_refine,
    }


def stateFloatAttrs(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack the per-vertex state into a single (V, C) float tensor so it can ride
    through `subdivideMarkedFaces(extra_vertex_attrs=...)` as ONE midpoint-blended
    attribute. Columns: [optimizable, blocked_count, resid_prev, resid_drop_ema,
    last_refine]. `resid_prev=inf` is encoded as a large finite sentinel so the
    midpoint average stays finite (a fresh midpoint has no history anyway)."""
    prev = state["resid_prev"]
    prev_enc = torch.where(torch.isfinite(prev), prev, torch.full_like(prev, 1e30))
    return torch.stack(
        [
            state["optimizable"].float(),
            state["blocked_count"].float(),
            prev_enc,
            state["resid_drop_ema"],
            state["last_refine"].float(),
        ],
        dim=1,
    )


def stateFromFloatAttrs(
    attrs: torch.Tensor, n_orig: int, device: str
) -> Dict[str, torch.Tensor]:
    """Inverse of `stateFloatAttrs` after subdivision blended the midpoints.

    Conservative re-quantization of the blended midpoint rows:
      optimizable : a new midpoint is optimizable iff EITHER parent was (blend
                    > 0), so refinement never silently freezes a fresh vertex.
      blocked_count: rounded blend, then forced to 0 for any optimizable vertex
                    (an optimizable vertex cannot already be "blocked").
      resid_prev  : decoded; the >=1e29 sentinel maps back to +inf (no history).
      last_refine : ceil of the blend (a midpoint is at least as refined as its
                    less-refined parent).
    Rows [0:n_orig] are the originals (their state is exact, not blended)."""
    optimizable = attrs[:, 0] > 0.5
    optimizable[:n_orig] = attrs[:n_orig, 0] > 0.5

    # midpoints: optimizable if either parent was -> blended value > 0
    mid_opt = attrs[n_orig:, 0] > 1e-6
    optimizable[n_orig:] = mid_opt

    blocked = attrs[:, 1].round().to(torch.int32)
    blocked[optimizable] = 0

    prev = attrs[:, 2]
    prev = torch.where(prev >= 1e29, torch.full_like(prev, float("inf")), prev)

    last_refine = torch.ceil(attrs[:, 4] - 1e-6).to(torch.int32)
    return {
        "optimizable": optimizable.to(device),
        "blocked_count": blocked.to(device),
        "resid_prev": prev.to(device),
        "resid_drop_ema": attrs[:, 3].to(device),
        "last_refine": last_refine.to(device),
    }
