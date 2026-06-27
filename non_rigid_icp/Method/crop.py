"""Axis-aligned bbox cropping of a triangle mesh into a compact sub-mesh.

Atomic and reusable: given a mesh and a centered axis-aligned box, keep the
faces that fall inside the box and re-index them into a compact
(vertices, faces) sub-mesh. Used to localize evaluation to a region of interest
(debug crops + region-restricted Chamfer/F1) without coupling to the fitter or
to any particular coordinate frame -- the caller passes whatever frame the box
is expressed in (here: the de-normalized / original-target frame the saved
result meshes live in).
"""

import numpy as np
from typing import Tuple, Union


def bboxFromCenterEdge(
    center: Union[np.ndarray, list, tuple],
    edge: Union[float, np.ndarray, list, tuple],
) -> Tuple[np.ndarray, np.ndarray]:
    """(lo, hi) corners of an axis-aligned box centered at `center`.

    `edge` is the full side length: a scalar (cube) or a per-axis (3,) vector.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    e = np.asarray(edge, dtype=np.float64).reshape(-1)
    half = (e if e.size == 3 else np.full(3, float(e.reshape(-1)[0]))) / 2.0
    return c - half, c + half


def verticesInBBox(
    vertices: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """(V,) bool mask of vertices inside the closed box [lo, hi]."""
    v = np.asarray(vertices)
    return ((v >= lo) & (v <= hi)).all(axis=1)


def cropMeshByBBox(
    vertices: np.ndarray,
    faces: np.ndarray,
    center: Union[np.ndarray, list, tuple],
    edge: Union[float, np.ndarray, list, tuple],
    mode: str = "all",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop a triangle mesh to an axis-aligned box, returning a compact sub-mesh.

    Args:
        vertices: (V, 3) float array.
        faces:    (F, 3) int array.
        center / edge: box center (3,) and full side length (scalar or (3,)).
        mode: which faces to keep --
            'all'      : every vertex of the face is inside the box (strict
                         containment, so the sub-mesh lies entirely in the box);
            'centroid' : the face centroid is inside (also keeps faces that
                         straddle the boundary).

    Returns:
        sub_vertices: (V', 3) the retained, re-indexed vertices.
        sub_faces:    (F', 3) re-indexed into `sub_vertices`.
        vert_keep:    (V,) bool, original vertices retained.
        face_keep:    (F,) bool, original faces retained.
    """
    v = np.asarray(vertices)
    f = np.asarray(faces).astype(np.int64)
    lo, hi = bboxFromCenterEdge(center, edge)

    if mode == "centroid":
        cen = v[f].mean(axis=1)
        face_keep = ((cen >= lo) & (cen <= hi)).all(axis=1)
    else:  # 'all' (strict containment)
        vin = verticesInBBox(v, lo, hi)
        face_keep = vin[f].all(axis=1)

    kept_faces = f[face_keep]
    vert_keep = np.zeros(v.shape[0], dtype=bool)
    if kept_faces.shape[0] > 0:
        vert_keep[kept_faces.reshape(-1)] = True

    remap = -np.ones(v.shape[0], dtype=np.int64)
    n_kept = int(vert_keep.sum())
    remap[vert_keep] = np.arange(n_kept)
    sub_vertices = v[vert_keep]
    if kept_faces.shape[0] > 0:
        sub_faces = remap[kept_faces]
    else:
        sub_faces = kept_faces.reshape(0, 3)
    return sub_vertices, sub_faces, vert_keep, face_keep


def cropMeshToBBoxUnion(
    vertices: np.ndarray,
    faces: np.ndarray,
    boxes,
    mode: str = "all",
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a mesh to the UNION of several axis-aligned boxes.

    A face is kept if it passes `cropMeshByBBox`'s `mode` test for ANY box, then
    the surviving faces are re-indexed once into a single compact sub-mesh. This
    is the atomic primitive behind the optional pre-fit crop: it shrinks a huge
    mesh to just the regions of interest while leaving each region exactly as the
    full-mesh run would see it (same box, same `mode`).

    Args:
        vertices: (V, 3) float array.
        faces:    (F, 3) int array.
        boxes:    iterable of (center(3,), edge(scalar|(3,))) pairs, all in the
                  SAME coordinate frame as `vertices`.
        mode:     'all' (strict containment) or 'centroid' (see cropMeshByBBox).

    Returns:
        sub_vertices: (V', 3) retained, re-indexed vertices.
        sub_faces:    (F', 3) re-indexed into `sub_vertices`.
    """
    v = np.asarray(vertices)
    f = np.asarray(faces).astype(np.int64)
    if f.shape[0] == 0:
        return v[:0], f.reshape(0, 3)

    face_keep = np.zeros(f.shape[0], dtype=bool)
    for center, edge in boxes:
        lo, hi = bboxFromCenterEdge(center, edge)
        if mode == "centroid":
            cen = v[f].mean(axis=1)
            keep = ((cen >= lo) & (cen <= hi)).all(axis=1)
        else:  # 'all' (strict containment)
            vin = verticesInBBox(v, lo, hi)
            keep = vin[f].all(axis=1)
        face_keep |= keep

    kept_faces = f[face_keep]
    vert_keep = np.zeros(v.shape[0], dtype=bool)
    if kept_faces.shape[0] > 0:
        vert_keep[kept_faces.reshape(-1)] = True
    remap = -np.ones(v.shape[0], dtype=np.int64)
    remap[vert_keep] = np.arange(int(vert_keep.sum()))
    sub_vertices = v[vert_keep]
    sub_faces = (
        remap[kept_faces] if kept_faces.shape[0] > 0 else kept_faces.reshape(0, 3)
    )
    return sub_vertices, sub_faces
