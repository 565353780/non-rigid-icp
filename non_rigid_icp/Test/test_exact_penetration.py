"""Unit tests for the exact triangle-triangle penetration predicate.

Verifies that `triangleTrianglePenetrate` (edge strictly through the other's
interior) flags TRUE penetration but NOT near-coplanar resting contact, while
the legacy Moller `triangleTriangleIntersects` over-counts the contact case.

Run (flux env, GPU 2):
  CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.test_exact_penetration
"""

import torch

from non_rigid_icp.Method.geometry import (
    triangleTrianglePenetrate,
    triangleTriangleIntersects,
    edgeTriangleCross,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _tri(pts):
    return torch.tensor([pts], dtype=torch.float32, device=DEV)


def test_true_penetration_flagged():
    """A vertical triangle stabbed through a horizontal one -> penetration."""
    horiz = _tri([[-1, -1, 0], [1, -1, 0], [0, 1, 0]])
    # vertical triangle whose base edge crosses the horizontal interior
    vert = _tri([[0, 0, -0.5], [0, 0, 0.5], [0.3, 0, 0.0]])
    pen = triangleTrianglePenetrate(vert, horiz)
    assert bool(pen[0]), "a clear stab-through must be flagged as penetration"
    print("[ok] test_true_penetration_flagged")


def test_coplanar_contact_not_penetration():
    """Two parallel triangles a hair apart (resting contact, no through-cross)
    must NOT be flagged by the exact predicate, even though Moller's coplanar
    fallback over-eagerly does for ~touching sheets."""
    bot = _tri([[0, 0, 0.0], [1, 0, 0.0], [0, 1, 0.0]])
    # an identical triangle slightly above: back-to-back sheets resting close.
    top = _tri([[0, 0, 1e-5], [1, 0, 1e-5], [0, 1, 1e-5]])
    pen = triangleTrianglePenetrate(bot, top)
    assert not bool(pen[0]), (
        "near-coplanar resting contact is NOT a through-penetration"
    )
    # Moller's coplanar/touch fallback (coplanar_eps default) flags it as a hit;
    # that is exactly the over-count we are eliminating.
    mol = triangleTriangleIntersects(bot, top)
    print(f"[ok] test_coplanar_contact_not_penetration "
          f"(exact={bool(pen[0])}, moller={bool(mol[0])})")


def test_boundary_touch_not_penetration():
    """An edge that just touches the other triangle's boundary (no interior
    crossing) is contact, not penetration."""
    horiz = _tri([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    # edge endpoint exactly on the horizontal plane at a boundary vertex
    touch = _tri([[0, 0, 0.0], [0, 0, 0.5], [0.5, 0.5, 0.5]])
    pen = triangleTrianglePenetrate(horiz, touch)
    assert not bool(pen[0]), "a boundary touch is not a strict penetration"
    print("[ok] test_boundary_touch_not_penetration")


def test_edge_cross_strictness():
    """edgeTriangleCross is strict: a segment ending exactly ON the triangle
    plane interior (t=1 boundary) is not a strict crossing."""
    tri_a = _tri([[-1, -1, 0], [1, -1, 0], [0, 1, 0]])[0]
    a, b, c = tri_a[0], tri_a[1], tri_a[2]
    # segment from below to exactly on the plane at the centroid (t=1 endpoint)
    p = torch.tensor([[0.0, -0.2, -0.5]], device=DEV)
    q = torch.tensor([[0.0, -0.2, 0.0]], device=DEV)  # lands ON plane
    a3 = a.unsqueeze(0); b3 = b.unsqueeze(0); c3 = c.unsqueeze(0)
    strict = edgeTriangleCross(p, q, a3, b3, c3, eps=0.0)
    assert not bool(strict[0]), "endpoint on the plane (t=1) is not strict cross"
    # push the endpoint through -> strict crossing
    q2 = torch.tensor([[0.0, -0.2, 0.2]], device=DEV)
    strict2 = edgeTriangleCross(p, q2, a3, b3, c3, eps=0.0)
    assert bool(strict2[0]), "passing through must be a strict crossing"
    print("[ok] test_edge_cross_strictness")


def main():
    test_true_penetration_flagged()
    test_coplanar_contact_not_penetration()
    test_boundary_touch_not_penetration()
    test_edge_cross_strictness()
    print("\nAll exact-penetration tests passed.")


if __name__ == "__main__":
    main()
