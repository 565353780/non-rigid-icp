"""Conforming local triangle subdivision.

Refines a marked subset of faces by edge-midpoint splitting. Every edge of a
marked face is split; faces incident to a split edge are re-triangulated
according to how many of their edges are split (1 -> 2, 2 -> 3, 3 -> 4 children).
Because the midpoint of a shared edge is one shared vertex, the result is
crack-free (no T-junctions) and keeps the original face winding. This is the
red-green style refinement restricted to a local region.

All operations are vectorized so the per-round cost is dominated by one global
unique-edge pass (the same one the fitter already performs).
"""

import torch
from typing import Tuple, List, Union

# child triangulation templates keyed by the 3-bit marked-edge code
# bit0 = edge(i0,i1), bit1 = edge(i1,i2), bit2 = edge(i2,i0)
# slot indices: 0,1,2 -> i0,i1,i2 ; 3,4,5 -> m0,m1,m2 (edge midpoints)
_TEMPLATES = {
    0: [(0, 1, 2)],
    1: [(0, 3, 2), (3, 1, 2)],
    2: [(1, 4, 0), (4, 2, 0)],
    4: [(2, 5, 1), (5, 0, 1)],
    3: [(3, 1, 4), (0, 3, 4), (0, 4, 2)],
    6: [(4, 2, 5), (1, 4, 5), (1, 5, 0)],
    5: [(5, 0, 3), (2, 5, 3), (2, 3, 1)],
    7: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],
}


def buildFaceEdgeIndex(
    triangles: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map each face's 3 edges to global unique-edge ids.

    Returns:
        unique_edges: (E, 2) long, sorted endpoints.
        face_edge_idx: (F, 3) long, columns = edges (i0,i1),(i1,i2),(i2,i0).
    """
    per_face = torch.stack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        dim=1,
    )  # (F, 3, 2)
    per_face = torch.sort(per_face, dim=2).values
    flat = per_face.reshape(-1, 2)
    unique_edges, inverse = torch.unique(flat, dim=0, return_inverse=True)
    face_edge_idx = inverse.reshape(-1, 3)
    return unique_edges, face_edge_idx


def subdivideMarkedFaces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    region_mask: torch.Tensor,
    extra_vertex_attrs: Union[List[torch.Tensor], None] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conformingly subdivide the faces in `region_mask`.

    Args:
        vertices: (V, 3) float tensor (the current/deformed base positions).
        faces: (F, 3) long tensor.
        region_mask: (F,) bool, faces to refine.
        extra_vertex_attrs: optional list of (V, C) tensors carried through the
            refinement (new midpoint rows = average of the two edge endpoints).
            Used to keep e.g. a clean-reference vertex field in lock-step with
            the deformed vertices so cumulative motion stays well-defined.

    Returns:
        new_vertices: (V + M, 3), original vertices unchanged, M new midpoints
            appended (position = edge midpoint, so the refinement is geometry
            preserving and the displacement can simply restart from zero).
        new_faces: (F', 3) long.
        parents: (F',) long, original face id each child came from.

    If `extra_vertex_attrs` is given, returns
    (new_vertices, new_faces, parents, new_extra_attrs_list) instead.
    """
    device = faces.device
    v = vertices.shape[0]
    f = faces.shape[0]

    if region_mask.sum() == 0:
        parents = torch.arange(f, device=device)
        if extra_vertex_attrs is not None:
            return vertices, faces, parents, list(extra_vertex_attrs)
        return vertices, faces, parents

    unique_edges, face_edge_idx = buildFaceEdgeIndex(faces)
    e = unique_edges.shape[0]

    marked_edge = torch.zeros(e, dtype=torch.bool, device=device)
    region_edge_ids = face_edge_idx[region_mask].reshape(-1)
    marked_edge[region_edge_ids] = True

    marked_idx = torch.nonzero(marked_edge, as_tuple=False).reshape(-1)
    n_marked = marked_idx.numel()

    midpoint_id = torch.full((e,), -1, dtype=torch.long, device=device)
    midpoint_id[marked_idx] = v + torch.arange(n_marked, device=device)

    e0 = unique_edges[marked_idx, 0]
    e1 = unique_edges[marked_idx, 1]
    mid_pos = 0.5 * (vertices[e0] + vertices[e1])
    new_vertices = torch.cat([vertices, mid_pos], dim=0)

    new_extra = None
    if extra_vertex_attrs is not None:
        new_extra = [
            torch.cat([attr, 0.5 * (attr[e0] + attr[e1])], dim=0)
            for attr in extra_vertex_attrs
        ]

    m = midpoint_id[face_edge_idx]  # (F, 3), -1 where edge unmarked
    me = marked_edge[face_edge_idx]  # (F, 3) bool
    code = (me[:, 0].long() + 2 * me[:, 1].long() + 4 * me[:, 2].long())

    slots = torch.cat([faces, m], dim=1)  # (F, 6)
    face_ids = torch.arange(f, device=device)

    new_faces_list = []
    parent_list = []
    for code_val, template in _TEMPLATES.items():
        sel = code == code_val
        if not bool(sel.any()):
            continue
        s = slots[sel]
        pid = face_ids[sel]
        for sa, sb, sc in template:
            child = torch.stack([s[:, sa], s[:, sb], s[:, sc]], dim=1)
            new_faces_list.append(child)
            parent_list.append(pid)

    new_faces = torch.cat(new_faces_list, dim=0)
    parents = torch.cat(parent_list, dim=0)
    if extra_vertex_attrs is not None:
        return new_vertices, new_faces, parents, new_extra
    return new_vertices, new_faces, parents
