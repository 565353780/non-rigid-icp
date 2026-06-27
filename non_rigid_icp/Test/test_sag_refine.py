"""Unit test for the sag-based subdivision criterion (error_field.faceSagError /
localizeSaggingFaces).

First principle being verified: a face whose three corners lie ON the target but
whose INTERIOR (centroid) tents off it (spanning a target ridge/valley) is the
only face worth refining. The test builds a target with a sharp valley and three
source faces:

  * face A: flat on the target plateau            -> sag ~ 0, NOT flagged.
  * face B: corners on the rims of the valley, big triangle spanning it so its
            centroid floats above the valley floor -> sag large, FLAGGED.
  * face C: small triangle entirely on the plateau -> sag ~ 0, NOT flagged.
"""

import numpy as np
import torch

from non_rigid_icp.Method.implicit_field import ImplicitField
from non_rigid_icp.Method.error_field import (
    faceCentroids,
    faceSagError,
    localizeSaggingFaces,
)
from non_rigid_icp.Method.topology import buildFaceAdjacency


def _build_target_with_valley():
    """A target surface: z=0 plateau everywhere except a deep V-shaped valley
    along the line y=0 (z dips to -0.5 at y=0). Triangulated finely so the
    closest-point field resolves the valley floor."""
    xs = np.linspace(-1.0, 1.0, 41)
    ys = np.linspace(-1.0, 1.0, 41)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    # deep narrow valley centered at y=0
    gz = -0.5 * np.exp(-(gy ** 2) / (2 * 0.06 ** 2))
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    nx, ny = xs.size, ys.size
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = a + 1
            c = a + ny
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.asarray(faces, dtype=np.int64)
    return verts, faces


def test_sag_localizer():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tV, tF = _build_target_with_valley()
    field = ImplicitField(tV, tF, device=dev)

    # source faces (corners chosen ON the target surface):
    #   A: flat on the plateau far from the valley (z=0)
    #   B: large triangle straddling the valley -- corners on the z=0 rims at
    #      y=+/-0.3 but centroid lands at y~0 where the target dips to ~-0.5
    #   C: small triangle on the plateau
    src_v = np.array([
        # face A (flat plateau)
        [-0.8, 0.5, 0.0], [-0.6, 0.5, 0.0], [-0.7, 0.7, 0.0],
        # face B: all 3 corners on the z=0 plateau at |y|>=0.35 (off the valley),
        # but arranged so the centroid lands at y~0 -> floats over the valley.
        [-0.3, -0.35, 0.0], [0.3, -0.35, 0.0], [0.0, 0.70, 0.0],
        # face C (small plateau)
        [0.6, 0.6, 0.0], [0.65, 0.6, 0.0], [0.625, 0.65, 0.0],
    ], dtype=np.float32)
    src_f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)

    verts = torch.tensor(src_v, device=dev)
    faces = torch.tensor(src_f, device=dev)
    adj = buildFaceAdjacency(faces)

    # sanity: corners are all on the target (vertex distance ~ 0)
    cp_v, _, _ = field.closestPoints(verts)
    vdist = (cp_v - verts).norm(dim=1)
    assert float(vdist.max()) < 1e-3, f"corners not on target: {vdist}"

    # centroid distances: A,C ~0, B large (floats over the valley floor)
    cents = faceCentroids(verts, faces)
    cp_c, _, _ = field.closestPoints(cents)
    cdist = (cp_c - cents).norm(dim=1)
    sag = faceSagError(vdist, cdist, faces)
    print("vertex dist:", vdist.cpu().numpy())
    print("centroid dist:", cdist.cpu().numpy())
    print("sag:", sag.cpu().numpy())

    assert sag[0] < 0.01, f"flat face A should not sag, got {sag[0]}"
    assert sag[2] < 0.01, f"small flat face C should not sag, got {sag[2]}"
    assert sag[1] > 0.1, f"valley-spanning face B should sag, got {sag[1]}"

    # localizer flags ONLY face B
    tau = 0.05
    region, sag2, stats = localizeSaggingFaces(
        verts, faces, field, adj, tau=tau,
        sag_mult=1.0, centroid_mult=1.0,
        min_component_faces=1, dilation_rings=0,
    )
    print("region:", region.cpu().numpy(), "stats:", stats)
    assert bool(region[1]) is True, "face B must be flagged"
    assert bool(region[0]) is False, "face A must NOT be flagged"
    assert bool(region[2]) is False, "face C must NOT be flagged"
    assert stats["n_sagging"] == 1
    print("[OK] test_sag_localizer")


def test_sag_self_terminating():
    """After splitting the sagging face once, the children's sag must shrink
    (their smaller centroids sit closer to the valley surface) -- demonstrating
    the criterion is self-terminating."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tV, tF = _build_target_with_valley()
    field = ImplicitField(tV, tF, device=dev)

    # the big valley-spanning triangle alone
    src_v = np.array([
        [0.0, -0.35, 0.0], [0.0, 0.35, 0.0], [0.5, 0.0, 0.0],
    ], dtype=np.float32)
    src_f = np.array([[0, 1, 2]], dtype=np.int64)
    verts = torch.tensor(src_v, device=dev)
    faces = torch.tensor(src_f, device=dev)

    cents = faceCentroids(verts, faces)
    cp_c, _, _ = field.closestPoints(cents)
    cdist0 = float((cp_c - cents).norm(dim=1)[0])

    # split into 4 by edge midpoints, PROJECT the new midpoints onto the target
    # (mimicking one optimization step), then re-measure children sag.
    e_ab = 0.5 * (verts[0] + verts[1])
    e_bc = 0.5 * (verts[1] + verts[2])
    e_ca = 0.5 * (verts[2] + verts[0])
    mids = torch.stack([e_ab, e_bc, e_ca], dim=0)
    cp_m, _, _ = field.closestPoints(mids)        # project midpoints onto target
    new_v = torch.cat([verts, cp_m], dim=0)
    # child faces: (0,3,5)(3,1,4)(5,4,2)(3,4,5)
    new_f = torch.tensor(
        [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]], device=dev
    )
    cents2 = faceCentroids(new_v, new_f)
    cp_c2, _, _ = field.closestPoints(cents2)
    cdist_children = (cp_c2 - cents2).norm(dim=1)
    print("parent centroid dist:", cdist0,
          "child centroid dists:", cdist_children.cpu().numpy())
    assert float(cdist_children.max()) < cdist0, (
        "children must sit closer to the target than the parent"
    )
    print("[OK] test_sag_self_terminating")


if __name__ == "__main__":
    test_sag_localizer()
    test_sag_self_terminating()
    print("ALL SAG TESTS PASSED")
