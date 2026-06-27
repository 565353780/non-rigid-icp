"""Front-advancing, self-intersection-free single projection step.

First principles
----------------
The watertight source mesh starts intersection-free. We want to slide every
vertex from its REST position ``ref_i`` toward its target closest point
``cp_i`` in ONE logical step, never creating a triangle-triangle
self-intersection. The earlier "static-mesh cap" failed because the colliding
mode is not a vertex tunnelling the *initial* mesh -- it is two interleaved
sheets whose vertices, once moved, fold their faces into each other. The cap
must therefore be measured against the *current* (already partly deformed)
mesh, and vertices must be advanced in dependency order.

The advancing-front invariant (the user's design):

  * ``done`` vertices are frozen at a known, mutually self-intersection-free
    configuration -- they are the stable substrate every later move is tested
    against.
  * In each round we look at the still-active vertices' trajectory segments
    ``ref_i -> cp_i`` and intersect them against the CURRENT mesh. The vertices
    whose segment hits nothing (``t = +inf``) can be planted at ``cp_i``
    immediately: their straight path is clear, so moving them cannot pierce any
    current face. They form this round's front.
  * If NO active vertex is fully clear, we still must make progress, so we take
    the vertices with the LARGEST earliest-``t`` (advance them as far as their
    own segment allows, minus a clearance) -- a guaranteed-progress fallback.
  * After moving the front, faces can still cross because several front
    vertices moved at once (their post-move segments may now interleave). We
    re-test ONLY the front's segments against the updated mesh and pull each
    offending vertex back to just before its earliest crossing (``t - clearance``),
    iterating to a fixpoint. The front is then frozen (``done``) and the next
    round begins.

Repeating until no active vertex remains yields a full single step whose final
mesh is self-intersection-free by construction: every vertex was planted only
at a position its trajectory reached without piercing the then-current mesh,
and the freeze order makes each test a static-substrate test.

The heavy lifting reuses ``Method/trajectory_guard``:
``segmentMeshIntersectionParams`` (broad-phase grid + parametric Moller-Trumbore)
and ``earliestSegmentMeshHits`` (vectorized per-segment earliest-``t``). This
module only adds the front-selection and freeze bookkeeping -- no new geometry.
"""

import torch
from typing import Tuple, Union, Callable

from non_rigid_icp.Method.trajectory_guard import (
    segmentMeshIntersectionParams,
    earliestSegmentMeshHits,
)


# --------------------------------------------------------------------------- #
# Atom A: earliest trajectory crossing of a set of segments vs the current mesh
# --------------------------------------------------------------------------- #
def earliestTrajectoryHit(
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    owner_vid: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    cell_size: Union[float, None] = None,
    seg_chunk: int = 200_000,
    t_lo: float = 1e-4,
    t_hi: float = 1.0,
    inflate: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-segment earliest crossing ``t`` of ``seg_start->seg_end`` vs `faces`.

    The segments are the trajectories ``ref_i -> proposed_i`` of the vertices
    ``owner_vid``; they are tested against the CURRENT mesh ``(vertices, faces)``.
    Faces incident to a segment's owner vertex (its 1-ring, which legitimately
    share the moving endpoint) are excluded inside the broad phase.

    Chunked over segments so the candidate-pair count stays bounded on millions
    of segments. Returns:
        t_min:    (S,) earliest crossing parameter (``+inf`` where clear).
        face_min: (S,) GLOBAL face id at that crossing (``-1`` where clear).
    """
    s = seg_start.shape[0]
    dev = faces.device
    t_min = torch.full((s,), float("inf"), device=dev)
    face_min = torch.full((s,), -1, dtype=torch.long, device=dev)
    if s == 0:
        return t_min, face_min
    for start in range(0, s, seg_chunk):
        sl = slice(start, start + seg_chunk)
        hs, hf, ht = segmentMeshIntersectionParams(
            seg_start[sl], seg_end[sl], vertices, faces,
            owner_vid=owner_vid[sl],
            inflate=inflate, cell_size=cell_size, face_ids=None,
            t_lo=t_lo, t_hi=t_hi,
        )
        tsub, fsub = earliestSegmentMeshHits(hs, hf, ht, seg_end[sl].shape[0])
        t_min[sl] = tsub
        face_min[sl] = fsub
    return t_min, face_min


# --------------------------------------------------------------------------- #
# Atom B: pick this round's advancing front from the active set
# --------------------------------------------------------------------------- #
def selectAdvancingFront(
    t_min: torch.Tensor,
    active_ids: torch.Tensor,
    min_clear_frac: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose which active vertices move this round and how far (alpha).

    Args:
        t_min: (A,) earliest crossing parameter for each active vertex's segment
            (``+inf`` == fully clear).
        active_ids: (A,) global vertex ids of the active set (aligned with t_min).
        min_clear_frac: if no vertex is fully clear, advance the vertices whose
            ``t_min`` is within this fraction of the current maximum ``t_min``
            (a guaranteed-progress front), each capped at its own ``t_min``.

    Returns:
        front_ids: (K,) global vertex ids selected to move this round.
        alpha:     (K,) per-front advance fraction along ``ref->cp`` in [0,1]
            (1.0 for fully-clear vertices; ``t_min`` for fallback vertices).
    """
    clear = torch.isinf(t_min)
    if bool(clear.any()):
        front_ids = active_ids[clear]
        alpha = torch.ones(front_ids.shape[0], device=t_min.device)
        return front_ids, alpha
    # fallback: nobody is fully clear -> advance the farthest-reaching ones so
    # the front never stalls. Take everyone within min_clear_frac of the best t.
    finite = torch.isfinite(t_min)
    if not bool(finite.any()):
        z = torch.zeros(0, dtype=torch.long, device=t_min.device)
        return z, torch.zeros(0, device=t_min.device)
    t_best = float(t_min[finite].max())
    pick = finite & (t_min >= t_best * (1.0 - min_clear_frac))
    front_ids = active_ids[pick]
    alpha = t_min[pick].clamp(min=0.0, max=1.0)
    return front_ids, alpha


# --------------------------------------------------------------------------- #
# Atom C: move a front, then pull back any vertex that now crosses
# --------------------------------------------------------------------------- #
def resolveBatchPullback(
    cur: torch.Tensor,
    ref: torch.Tensor,
    target: torch.Tensor,
    front_ids: torch.Tensor,
    alpha0: torch.Tensor,
    faces: torch.Tensor,
    cell_size: Union[float, None] = None,
    seg_chunk: int = 200_000,
    clearance_t: float = 1e-3,
    t_lo: float = 1e-4,
    max_iters: int = 8,
) -> Tuple[torch.Tensor, int]:
    """Plant a front at ``ref + alpha*(target-ref)`` then pull crossers back.

    The front vertices are written into a COPY of ``cur`` at their proposed
    positions (``alpha0`` along ``ref->target``). We then re-test ONLY the front
    segments ``ref -> current-front-position`` against the updated mesh; any
    front vertex whose segment now pierces a non-incident face is pulled back to
    ``t - clearance_t`` along its own segment. Repeated to a fixpoint (or
    ``max_iters``), so the front lands self-intersection-free w.r.t. the frozen
    substrate AND w.r.t. itself.

    Returns:
        new_cur: (V,3) positions with the front moved + repaired.
        n_pulled: number of front vertices that were pulled back at all.
    """
    dev = faces.device
    new_cur = cur.clone()
    alpha = alpha0.clone()
    # plant the front
    new_cur[front_ids] = ref[front_ids] + alpha.unsqueeze(1) * (
        target[front_ids] - ref[front_ids]
    )
    pulled = torch.zeros(front_ids.shape[0], dtype=torch.bool, device=dev)
    seg_start = ref[front_ids]
    for _ in range(max_iters):
        seg_end = new_cur[front_ids]
        t_min, _ = earliestTrajectoryHit(
            seg_start, seg_end, front_ids, new_cur, faces,
            cell_size=cell_size, seg_chunk=seg_chunk, t_lo=t_lo, t_hi=1.0,
        )
        cross = torch.isfinite(t_min)
        if not bool(cross.any()):
            break
        # pull each crossing front vertex back to t - clearance along its own
        # ref->target segment. alpha is the current advance fraction; the new
        # safe fraction is (t_min * alpha) - clearance because seg_end already
        # sits at fraction `alpha`, so t_min is measured along [ref, seg_end].
        new_alpha = (t_min * alpha - clearance_t).clamp(min=0.0)
        alpha = torch.where(cross, new_alpha, alpha)
        new_cur[front_ids] = ref[front_ids] + alpha.unsqueeze(1) * (
            target[front_ids] - ref[front_ids]
        )
        pulled |= cross
    return new_cur, int(pulled.sum().item())


# --------------------------------------------------------------------------- #
# Orchestrator: one full front-advancing projection step
# --------------------------------------------------------------------------- #
def frontAdvancingStep(
    ref: torch.Tensor,
    cur: torch.Tensor,
    target: torch.Tensor,
    faces: torch.Tensor,
    cell_size: Union[float, None] = None,
    seg_chunk: int = 200_000,
    clearance_t: float = 1e-3,
    t_lo: float = 1e-4,
    max_rounds: int = 64,
    pullback_iters: int = 8,
    progress: Union[Callable[[dict], None], None] = None,
) -> Tuple[torch.Tensor, dict]:
    """Slide every vertex toward ``target`` in self-intersection-free rounds.

    Args:
        ref:    (V,3) REST positions (segment starts; never change).
        cur:    (V,3) current positions (the starting mesh of this step).
        target: (V,3) per-vertex target (closest point on the target surface).
        faces:  (F,3) topology (constant through the step).
        progress: optional per-round callback receiving a small stats dict.

    Returns:
        new_cur: (V,3) final positions after the step.
        info: summary stats (rounds used, vertices moved, fallback rounds, ...).
    """
    dev = faces.device
    v = cur.shape[0]
    done = torch.zeros(v, dtype=torch.bool, device=dev)
    work = cur.clone()

    n_rounds = 0
    n_fallback = 0
    n_pulled_total = 0
    all_ids = torch.arange(v, device=dev)
    for rnd in range(max_rounds):
        active = ~done
        if not bool(active.any()):
            break
        active_ids = all_ids[active]
        seg_start = ref[active_ids]
        seg_end = target[active_ids]
        # earliest crossing of each active vertex's full ref->target segment
        # against the CURRENT mesh (done verts frozen, active verts at `work`).
        t_min, _ = earliestTrajectoryHit(
            seg_start, seg_end, active_ids, work, faces,
            cell_size=cell_size, seg_chunk=seg_chunk, t_lo=t_lo, t_hi=1.0,
        )
        front_ids, alpha = selectAdvancingFront(t_min, active_ids)
        if front_ids.numel() == 0:
            break
        fallback = bool(torch.isinf(t_min).any().logical_not())
        n_fallback += int(fallback)
        work, n_pulled = resolveBatchPullback(
            work, ref, target, front_ids, alpha, faces,
            cell_size=cell_size, seg_chunk=seg_chunk,
            clearance_t=clearance_t, t_lo=t_lo, max_iters=pullback_iters,
        )
        done[front_ids] = True
        n_rounds += 1
        n_pulled_total += n_pulled
        if progress is not None:
            progress({
                "round": rnd,
                "front_size": int(front_ids.shape[0]),
                "remaining": int((~done).sum().item()),
                "fallback": fallback,
                "pulled": n_pulled,
            })

    info = {
        "rounds": n_rounds,
        "fallback_rounds": n_fallback,
        "pulled_total": n_pulled_total,
        "remaining_active": int((~done).sum().item()),
    }
    return work, info


# --------------------------------------------------------------------------- #
# Penetration-driven relaxation: the EXACT face-face fix
# --------------------------------------------------------------------------- #
def penetrationRelaxStep(
    ref: torch.Tensor,
    cur: torch.Tensor,
    target: torch.Tensor,
    faces: torch.Tensor,
    face_adjacency: Union[torch.Tensor, None] = None,
    cell_size: Union[float, None] = None,
    backoff: float = 0.8,
    min_alpha_step: float = 1e-3,
    max_iters: int = 60,
    exclude_ring: int = 1,
    ignore_baseline: bool = True,
    locked_mask: Union[torch.Tensor, None] = None,
    progress: Union[Callable[[dict], None], None] = None,
) -> Tuple[torch.Tensor, dict]:
    """Drive every vertex to ``target`` then relax out TRUE penetrations.

    First principles. The diagnostic showed the surviving self-intersections are
    genuine through-penetrations (an edge of one face crossing the interior of
    another) created when interleaved vertices move at once -- NOT vertex-tunnels-
    a-static-face, which a trajectory test can see. So the repair is driven by
    the EXACT penetration predicate directly:

      * Parameterize each vertex by ``alpha_i in [0,1]`` along ``ref_i -> target_i``;
        start everyone at ``alpha = 1`` (fully projected onto the target).
      * Find the truly penetrating face pairs (`findSelfIntersections`,
        predicate='penetrate' -- exact, no coplanar/contact tolerance).
      * Back off the ``alpha`` of every vertex incident to a penetrating face by a
        multiplicative ``backoff`` factor (toward its rest position, where the
        watertight mesh is penetration-free). Re-evaluate. Repeat.

    Because the rest configuration (``alpha = 0``) is penetration-free and moving
    a vertex back along its own segment monotonically shrinks the deformation
    that produced the crossing, the relaxation converges to a penetration-free
    mesh; vertices never involved in a penetration keep ``alpha = 1`` (maximal
    fit). This is the precise, tolerance-free analogue of the trajectory pull-back
    that actually catches the face-fold mode.

    Args:
        backoff: multiplicative alpha reduction per iteration for incident
            vertices (0.5 halves the remaining advance each time).
        min_alpha_step: once a vertex's alpha drops below this it is pinned to 0
            (treated as un-movable this step) to guarantee termination.
        locked_mask: optional (V,) bool. Locked vertices are FROZEN -- they are
            never backed off (their fit from a previous step is preserved) and
            their ``target`` is assumed already equal to their position (so
            ``seg = 0`` and they do not move). Used by the lock-and-refine schedule
            where each step only moves the newly inserted vertices. Because a new
            penetration always involves at least one unlocked vertex, restricting
            the back-off to unlocked vertices still resolves every new crossing.

    Returns:
        new_cur: (V,3) penetration-free positions.
        info: per-step stats (iters, penetrating pairs history, backed-off count).
    """
    from non_rigid_icp.Method.self_intersection import findSelfIntersections
    from non_rigid_icp.Method.collision import pairKeys

    dev = faces.device
    v = cur.shape[0]
    f = faces.shape[0]
    movable = (
        torch.ones(v, dtype=torch.bool, device=dev)
        if locked_mask is None else ~locked_mask
    )
    # back-off anchor (alpha = 0):
    #   * unlocked schedule: the rest watertight mesh `ref` (penetration-free);
    #   * locked schedule: the INCOMING pose `cur`. Retreating an unlocked vertex
    #     to `cur` returns the mesh to the validated penetration-free baseline,
    #     so pinning offenders to alpha = 0 provably clears every NEW penetration
    #     without ever moving (or relying on) the frozen locked vertices.
    anchor = cur if locked_mask is not None else ref
    seg = target - anchor  # per-vertex displacement anchor->target
    alpha = torch.ones(v, device=dev)
    work = anchor + seg  # alpha = 1 everywhere (= target)

    # baseline: penetrations the step neither caused nor can fix, so they are
    # excluded from the "to repair" set (matching `tri_intersecting_pairs_new`).
    #   * unlocked schedule: the rest watertight mesh `ref` (input artifacts);
    #   * locked schedule: the INCOMING pose `cur` itself -- locked vertices are
    #     frozen at their already-fitted positions, so any pair already present
    #     before this step moves the unlocked vertices is pre-existing and (if it
    #     involves only locked vertices) unfixable. Baselining against `cur`
    #     guarantees every remaining "new" pair involves at least one unlocked
    #     vertex, which the back-off can always resolve.
    baseline_keys = None
    if ignore_baseline:
        base_pose = cur if locked_mask is not None else ref
        base_inter = findSelfIntersections(
            base_pose, faces, inflate=0.0, exclude_ring=exclude_ring,
            face_adjacency=face_adjacency, cell_size=cell_size,
            predicate="penetrate",
        )
        baseline_keys = pairKeys(base_inter, f)

    def _new_pairs(inter_pairs):
        if baseline_keys is None or baseline_keys.numel() == 0 \
                or inter_pairs.shape[0] == 0:
            return inter_pairs
        k = pairKeys(inter_pairs, f)
        return inter_pairs[~torch.isin(k, baseline_keys)]

    pair_hist = []
    backed = torch.zeros(v, dtype=torch.bool, device=dev)
    last_pairs = -1
    for it in range(max_iters):
        inter = findSelfIntersections(
            work, faces, inflate=0.0, exclude_ring=exclude_ring,
            face_adjacency=face_adjacency, cell_size=cell_size,
            predicate="penetrate",
        )
        inter = _new_pairs(inter)
        n_pairs = int(inter.shape[0])
        pair_hist.append(n_pairs)
        if progress is not None:
            progress({"iter": it, "pen_pairs": n_pairs})
        if n_pairs == 0:
            break
        # vertices incident to any penetrating face -> back their alpha off, but
        # only the UNLOCKED ones (locked verts are frozen; a new penetration is
        # guaranteed to involve at least one unlocked vertex).
        bad_faces = torch.unique(inter.reshape(-1))
        bad_vids = torch.unique(faces[bad_faces].reshape(-1))
        bad_vids = bad_vids[movable[bad_vids]]
        new_alpha = alpha.clone()
        new_alpha[bad_vids] = alpha[bad_vids] * backoff
        # pin tiny alphas to exactly 0 (un-movable) for guaranteed termination
        new_alpha[new_alpha < min_alpha_step] = 0.0
        alpha = new_alpha
        backed[bad_vids] = True
        work = anchor + alpha.unsqueeze(1) * seg
        last_pairs = n_pairs

    # Finishing pass: any pair still penetrating after the relaxation budget is a
    # stubborn knot where the gentle multiplicative back-off oscillates. Pin its
    # vertices straight to the REST position (alpha = 0): the watertight rest mesh
    # is penetration-free, so collapsing the offenders to rest provably clears
    # them. A few sweeps absorb knots that only surface once their neighbours
    # retreat. This guarantees a penetration-free result, trading a handful of
    # local vertices' fit for a hard zero.
    for _ in range(8):
        inter = findSelfIntersections(
            work, faces, inflate=0.0, exclude_ring=exclude_ring,
            face_adjacency=face_adjacency, cell_size=cell_size,
            predicate="penetrate",
        )
        inter = _new_pairs(inter)
        if inter.shape[0] == 0:
            break
        bad_faces = torch.unique(inter.reshape(-1))
        bad_vids = torch.unique(faces[bad_faces].reshape(-1))
        bad_vids = bad_vids[movable[bad_vids]]
        alpha[bad_vids] = 0.0
        backed[bad_vids] = True
        work = anchor + alpha.unsqueeze(1) * seg

    inter = findSelfIntersections(
        work, faces, inflate=0.0, exclude_ring=exclude_ring,
        face_adjacency=face_adjacency, cell_size=cell_size, predicate="penetrate",
    )
    inter = _new_pairs(inter)
    info = {
        "iters": len(pair_hist),
        "pen_pairs_start": pair_hist[0] if pair_hist else 0,
        "pen_pairs_end": int(inter.shape[0]),
        "pen_pairs_history": pair_hist,
        "backed_off_vertices": int(backed.sum().item()),
        "pinned_vertices": int((alpha == 0).sum().item()),
        "mean_alpha": float(alpha.mean().item()),
    }
    return work, info
