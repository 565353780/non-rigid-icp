"""Unit tests for the front-advancing self-intersection-free step atoms.

Each atom is checked on tiny, hand-verifiable meshes; then the orchestrator is
run on an interleaved two-strip configuration (the synthetic analogue of the
double-sheet collapse) and the result is asserted self-intersection-free by the
authoritative `findSelfIntersections` scan.

Run (flux env, GPU 2):
  CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.test_front_advance
"""

import torch

from non_rigid_icp.Method.front_advance import (
    earliestTrajectoryHit,
    selectAdvancingFront,
    resolveBatchPullback,
    frontAdvancingStep,
    penetrationRelaxStep,
)
from non_rigid_icp.Method.self_intersection import findSelfIntersections


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _quad(z, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    """Two triangles forming a quad at height z. Returns (verts(4,3), faces(2,3))."""
    v = torch.tensor(
        [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
        dtype=torch.float32, device=DEV,
    )
    f = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=DEV)
    return v, f


def test_atom_a_hit_and_clear():
    """A segment that passes through a quad hits it (t in (0,1)); one that
    stops short is clear (t=+inf)."""
    quad_v, quad_f = _quad(0.5)  # a horizontal sheet at z=0.5
    # extra vertex 4 starts below, segment goes up through the sheet
    verts = torch.cat([quad_v, torch.tensor([[0.5, 0.5, 0.0]],
                                            dtype=torch.float32, device=DEV)], 0)
    faces = quad_f
    seg_start = verts[4:5]
    seg_end = torch.tensor([[0.5, 0.5, 1.0]], device=DEV)  # crosses z=0.5 at t=0.5
    owner = torch.tensor([4], device=DEV)
    t_min, face_min = earliestTrajectoryHit(seg_start, seg_end, owner, verts, faces)
    assert torch.isfinite(t_min[0]), "should detect the crossing"
    assert abs(float(t_min[0]) - 0.5) < 1e-3, f"t should be ~0.5, got {t_min[0]}"

    seg_end2 = torch.tensor([[0.5, 0.5, 0.4]], device=DEV)  # stops below sheet
    t_min2, _ = earliestTrajectoryHit(seg_start, seg_end2, owner, verts, faces)
    assert torch.isinf(t_min2[0]), "a segment stopping short must be clear"
    print("[ok] test_atom_a_hit_and_clear")


def test_atom_b_front_selection():
    """Fully-clear (inf) vertices are picked with alpha=1; if none clear, the
    farthest-t vertices are picked with alpha=t."""
    ids = torch.tensor([10, 11, 12, 13], device=DEV)
    t_inf = torch.tensor([float("inf"), 0.3, float("inf"), 0.7], device=DEV)
    front, alpha = selectAdvancingFront(t_inf, ids)
    assert set(front.tolist()) == {10, 12}, front.tolist()
    assert torch.allclose(alpha, torch.ones_like(alpha)), alpha

    t_none = torch.tensor([0.2, 0.3, 0.9, 0.88], device=DEV)
    front2, alpha2 = selectAdvancingFront(t_none, ids, min_clear_frac=0.05)
    # best t = 0.9; within 5% -> >=0.855 -> picks 0.9 and 0.88 (ids 12,13)
    assert set(front2.tolist()) == {12, 13}, front2.tolist()
    assert torch.allclose(alpha2, torch.tensor([0.9, 0.88], device=DEV)), alpha2
    print("[ok] test_atom_b_front_selection")


def test_atom_c_pullback_stops_before_sheet():
    """Moving a vertex straight through a frozen sheet must be pulled back to
    just before the sheet (alpha ~ t_cross - clearance)."""
    quad_v, quad_f = _quad(0.5)
    moving = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32, device=DEV)
    verts = torch.cat([quad_v, moving], 0)
    faces = quad_f
    ref = verts.clone()
    target = verts.clone()
    target[4] = torch.tensor([0.5, 0.5, 1.0], device=DEV)  # wants to cross to z=1
    front_ids = torch.tensor([4], device=DEV)
    alpha0 = torch.ones(1, device=DEV)
    new_cur, n_pulled = resolveBatchPullback(
        verts, ref, target, front_ids, alpha0, faces, clearance_t=1e-3,
    )
    z_final = float(new_cur[4, 2])
    assert n_pulled == 1, "the crossing vertex must be pulled back"
    assert z_final < 0.5, f"must stop below the sheet z=0.5, got {z_final}"
    assert z_final > 0.49, f"must stop JUST below (min pull-back), got {z_final}"
    print(f"[ok] test_atom_c_pullback_stops_before_sheet (z={z_final:.4f})")


def _interleaved_double_strip():
    """Two thin strips whose targets cross each other -- the synthetic analogue
    of the double-sheet collapse. Returns ref, target, faces."""
    # bottom strip at z=0, top strip at z=0.2; their targets swap sides so a
    # naive simultaneous move interleaves their faces.
    n = 6
    xs = torch.linspace(0, 1, n, device=DEV)
    bot = torch.stack([xs, torch.zeros(n, device=DEV),
                       torch.zeros(n, device=DEV)], 1)
    bot2 = bot + torch.tensor([0.0, 0.2, 0.0], device=DEV)
    top = bot + torch.tensor([0.0, 0.0, 0.2], device=DEV)
    top2 = top + torch.tensor([0.0, 0.2, 0.0], device=DEV)
    ref = torch.cat([bot, bot2, top, top2], 0)  # 4n verts

    def strip_faces(a, b, n):
        f = []
        for i in range(n - 1):
            f.append([a + i, a + i + 1, b + i + 1])
            f.append([a + i, b + i + 1, b + i])
        return f

    faces = []
    faces += strip_faces(0, n, n)        # bottom strip
    faces += strip_faces(2 * n, 3 * n, n)  # top strip
    faces = torch.tensor(faces, dtype=torch.long, device=DEV)

    # targets: push both strips toward z=0.1 (they converge) -- the collapse.
    target = ref.clone()
    target[:, 2] = 0.1
    return ref, target, faces


def test_orchestrator_no_self_intersection():
    ref, target, faces = _interleaved_double_strip()
    cur = ref.clone()
    # naive simultaneous projection (the failing baseline): move all to target
    naive = target.clone()
    naive_inter = findSelfIntersections(naive, faces, exclude_ring=1)

    new_cur, info = frontAdvancingStep(
        ref, cur, target, faces, clearance_t=1e-3, t_lo=1e-4,
    )
    inter = findSelfIntersections(new_cur, faces, exclude_ring=1)
    print(f"[info] naive intersections={naive_inter.shape[0]}, "
          f"front-advancing intersections={inter.shape[0]}, info={info}")
    assert inter.shape[0] == 0, (
        f"front-advancing step must be self-intersection-free, "
        f"got {inter.shape[0]} pairs"
    )
    print("[ok] test_orchestrator_no_self_intersection")


def _crossing_strips_true_penetration():
    """Two independent triangles (6 verts, 2 faces) clearly separated at REST;
    their TARGETS make the vertical triangle stab straight through the
    horizontal one's interior -- a hand-verifiable true through-penetration.
    Backing off toward rest (separated) provably removes it. Returns ref,
    target, faces. The two faces share no vertex, so exclude_ring=1 keeps them."""
    # rest: horizontal triangle at z=0; vertical triangle sitting well above it
    ref = torch.tensor([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],   # horiz (face 0)
        [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.3, 0.0, 1.5],      # vert  (face 1)
    ], dtype=torch.float32, device=DEV)
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long, device=DEV)
    # target: drop the vertical triangle so its base edge crosses z=0 interior
    target = ref.clone()
    target[3] = torch.tensor([0.0, 0.0, -0.5], device=DEV)
    target[4] = torch.tensor([0.0, 0.0, 0.5], device=DEV)
    target[5] = torch.tensor([0.3, 0.0, 0.0], device=DEV)
    return ref, target, faces


def test_penetration_relax_clears_true_penetration():
    """A configuration whose naive projection produces a genuine through-
    penetration must be relaxed to exactly zero by the penetration-driven step,
    using the EXACT predicate. We assert the relax loop actually engages
    (backs vertices off) and ends penetration-free."""
    ref, target, faces = _crossing_strips_true_penetration()
    naive = findSelfIntersections(target, faces, exclude_ring=1,
                                  predicate="penetrate")
    new_cur, info = penetrationRelaxStep(
        ref, ref.clone(), target, faces, backoff=0.5, max_iters=40,
    )
    inter = findSelfIntersections(new_cur, faces, exclude_ring=1,
                                  predicate="penetrate")
    print(f"[info] naive true-penetrations={naive.shape[0]}, "
          f"after relax={inter.shape[0]}, info={info}")
    assert naive.shape[0] > 0, (
        "test setup must actually create a true penetration to be meaningful; "
        f"got {naive.shape[0]} -- adjust the synthetic case"
    )
    assert inter.shape[0] == 0, (
        f"penetration-relax must remove all TRUE penetrations, "
        f"got {inter.shape[0]}"
    )
    assert info["backed_off_vertices"] > 0, "relax loop must have engaged"
    print("[ok] test_penetration_relax_clears_true_penetration")


def test_penetration_relax_keeps_clean_at_full_fit():
    """When the projection is already penetration-free, the relax step must not
    back anything off -- every vertex stays at alpha=1 (full fit)."""
    # a single clean strip moved to a clearly non-crossing target
    n = 5
    xs = torch.linspace(0, 1, n, device=DEV)
    bot = torch.stack([xs, torch.zeros(n, device=DEV),
                       torch.zeros(n, device=DEV)], 1)
    bot2 = bot + torch.tensor([0.0, 0.2, 0.0], device=DEV)
    ref = torch.cat([bot, bot2], 0)
    faces = []
    for i in range(n - 1):
        faces.append([i, i + 1, n + i + 1])
        faces.append([i, n + i + 1, n + i])
    faces = torch.tensor(faces, dtype=torch.long, device=DEV)
    target = ref + torch.tensor([0.0, 0.0, 0.05], device=DEV)  # flat lift
    new_cur, info = penetrationRelaxStep(ref, ref.clone(), target, faces)
    assert info["pen_pairs_end"] == 0, info
    assert abs(info["mean_alpha"] - 1.0) < 1e-6, (
        f"clean fit must keep alpha=1, got mean_alpha={info['mean_alpha']}"
    )
    assert torch.allclose(new_cur, target, atol=1e-6), "must land on target"
    print("[ok] test_penetration_relax_keeps_clean_at_full_fit")


def main():
    test_atom_a_hit_and_clear()
    test_atom_b_front_selection()
    test_atom_c_pullback_stops_before_sheet()
    test_orchestrator_no_self_intersection()
    test_penetration_relax_clears_true_penetration()
    test_penetration_relax_keeps_clean_at_full_fit()
    print("\nAll front-advance tests passed.")


if __name__ == "__main__":
    main()
