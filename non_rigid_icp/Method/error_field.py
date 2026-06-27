"""Localize high fitting-error regions on the source mesh.

Combines the two error directions of the fit:
  fit error      source -> target : where the deformed source sits off target.
  coverage error target -> source : where the original target is under-covered.

Both are reduced to a per-face error, then thresholded (relative to tau and a
high quantile) and cleaned into connected, dilated face regions that the local
subdivision will refine. All steps reuse the topology / NN atoms.
"""

import torch
import numpy as np
from typing import Tuple, Union

from typing import Optional

from non_rigid_icp.Method.nn import NNIndex
from non_rigid_icp.Method.topology import (
    dilateFaceMask,
    connectedFaceComponents,
)
from non_rigid_icp.Method.fit_state import faceStateFromVertexState


def faceErrorFromVertexError(
    vertex_error: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Per-face error as the mean of its three vertex errors. (F,)"""
    return vertex_error[faces].mean(dim=1)


def faceCentroids(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-face centroid (mean of the three vertices). (F, 3)."""
    return vertices[faces].mean(dim=1)


def faceSagError(
    vertex_dist: torch.Tensor,
    centroid_dist: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """The "sag" of every face = how far its INTERIOR sits off the target
    surface beyond what its three corners already account for:

        sag(f) = d(centroid(f), target) - mean_i d(v_i, target)

    First principles: the goal is to put the WHOLE source surface on the target.
    A vertex's distance is driven to ~0 by the per-step closest-point projection,
    so a face with all three corners on the target but a far-off centroid is a
    flat triangle *tented* across a target feature (ridge / valley / hole edge) --
    its interior is the only part still off-surface. `sag` isolates exactly that
    defect and is ~0 for a face that already lies flat on the target (corners and
    centroid equidistant), so refining by `sag` is self-terminating: splitting a
    tented face inserts midpoints that the next projection pulls onto the target,
    shrinking the children's sag until they fall within tolerance. Refining any
    OTHER face cannot lower the surface-to-surface error and only multiplies face
    count -- hence `sag` gives "the fewest faces for the lowest error".

    `vertex_dist` is (V,), `centroid_dist` is (F,). Returns (F,) (clamped >= 0).
    """
    mean_vert = vertex_dist[faces].mean(dim=1)
    return (centroid_dist - mean_vert).clamp(min=0.0)


def localizeSaggingFaces(
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    field,
    face_adjacency: torch.Tensor,
    tau: float,
    *,
    sag_mult: float = 1.0,
    centroid_mult: float = 1.0,
    quantile: float = 0.0,
    min_component_faces: int = 1,
    dilation_rings: int = 0,
    max_faces: Union[int, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Locate faces whose INTERIOR sags off the target while their corners are
    already on it -- the only faces worth subdividing for the "whole surface on
    target" goal (see `faceSagError`).

    Uses the exact closest-point `field` (an `ImplicitField`) to measure the
    true distance of both the vertices and the face centroids to the target
    surface (face-interior accurate, unlike a sampled-point k-NN), then keeps a
    face when BOTH:
        sag(f)         > sag_mult      * tau   (interior tented beyond corners)
        d(centroid, T) > centroid_mult * tau   (and the interior is itself out of
                                                tolerance -- a face already within
                                                tau everywhere needs no split).
    `quantile` (optional) raises the sag bar to the worst (1-quantile) fraction;
    `max_faces` caps the per-round budget to the highest-sag faces. All batched,
    no Python loops over faces.

    Returns (region_mask (F,) bool, sag (F,) float, stats dict).
    """
    device = deformed_vertices.device
    # exact distance of every vertex and every face centroid to the target.
    cp_v, _, _ = field.closestPoints(deformed_vertices)
    vert_dist = (cp_v - deformed_vertices).norm(dim=1)              # (V,)
    centroids = faceCentroids(deformed_vertices, faces)            # (F,3)
    cp_c, _, _ = field.closestPoints(centroids)
    cent_dist = (cp_c - centroids).norm(dim=1)                     # (F,)

    sag = faceSagError(vert_dist, cent_dist, faces)               # (F,)

    sag_thr = sag_mult * float(tau)
    if quantile and quantile > 0.0:
        if sag.numel() < 8_000_000:
            sample = sag.float()
        else:
            perm = torch.randperm(sag.numel(), device=device)[:8_000_000]
            sample = sag.float()[perm]
        sag_thr = max(sag_thr, torch.quantile(sample, quantile).item())

    raw_mask = (sag > sag_thr) & (cent_dist > centroid_mult * float(tau))

    if max_faces is not None and int(raw_mask.sum().item()) > max_faces:
        masked = torch.where(raw_mask, sag, torch.full_like(sag, -1.0))
        topk = torch.topk(masked, max_faces).indices
        capped = torch.zeros_like(raw_mask)
        capped[topk] = True
        raw_mask = capped

    # optional clean of tiny components, then optional dilation for crack-free
    # children. Defaults (min_component_faces=1, dilation_rings=0) keep ONLY the
    # truly sagging faces so the face budget is minimal; the conforming
    # subdivision already inserts a transition fan so no crack appears.
    if min_component_faces > 1:
        labels, sizes = connectedFaceComponents(raw_mask, face_adjacency)
        if sizes.size > 0:
            small = np.nonzero(sizes < min_component_faces)[0]
            if small.size > 0:
                cleaned_np = raw_mask.detach().cpu().numpy().copy()
                cleaned_np[np.isin(labels, small)] = False
                raw_mask = torch.from_numpy(cleaned_np).to(device)
    region_mask = (
        dilateFaceMask(raw_mask, face_adjacency, dilation_rings)
        if dilation_rings > 0 else raw_mask
    )

    stats = {
        "tau": float(tau),
        "sag_threshold": float(sag_thr),
        "sag_max": float(sag.max().item()) if sag.numel() else 0.0,
        "sag_mean": float(sag.mean().item()) if sag.numel() else 0.0,
        "centroid_dist_max": float(cent_dist.max().item()) if cent_dist.numel() else 0.0,
        "vert_dist_mean": float(vert_dist.mean().item()) if vert_dist.numel() else 0.0,
        "n_sagging": int(raw_mask.sum().item()),
        "n_region": int(region_mask.sum().item()),
    }
    return region_mask, sag, stats


def fitVertexError(
    deformed_vertices: torch.Tensor, target_index: NNIndex
) -> torch.Tensor:
    """Per source-vertex distance to the nearest target surface point. (V,)"""
    idx, d2 = target_index.query(deformed_vertices, k=1)
    d = torch.from_numpy(d2).to(deformed_vertices.device).clamp(min=0.0).sqrt()
    return d


def coverageVertexError(
    deformed_vertices: torch.Tensor,
    target_points: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Coverage error per source vertex.

    For each target point we find the nearest source vertex and record the
    distance there (worst-case). Source vertices adjacent to large uncovered
    target areas get a high value. Returns (V,).
    """
    v = deformed_vertices.shape[0]
    src_np = deformed_vertices.detach().cpu().numpy().astype(np.float32)
    src_index = NNIndex(src_np, device=device)
    idx, d2 = src_index.query(target_points, k=1)
    idx_t = torch.from_numpy(idx).to(deformed_vertices.device)
    d_t = torch.from_numpy(d2).to(deformed_vertices.device).clamp(min=0.0).sqrt()

    cov = torch.zeros(v, device=deformed_vertices.device)
    cov.scatter_reduce_(0, idx_t, d_t, reduce="amax", include_self=True)
    return cov


def selectHighErrorFaces(
    face_error: torch.Tensor,
    tau: float,
    error_mult: float = 2.0,
    quantile: float = 0.9,
    max_faces: Union[int, None] = None,
) -> Tuple[torch.Tensor, float]:
    """Boolean mask of faces to refine.

    The primary criterion is ABSOLUTE: refine faces whose error exceeds the
    tolerance `error_mult * tau` (faces already within tolerance need no
    refinement). The `quantile` acts as an upper bound (refine at most the worst
    (1 - quantile) fraction) and `max_faces` as a hard cap, both of which keep
    the per-round face growth bounded on very large meshes.

    Returns (mask, threshold).
    """
    if face_error.numel() < 8_000_000:
        sample = face_error.float()
    else:
        perm = torch.randperm(face_error.numel(), device=face_error.device)[:8_000_000]
        sample = face_error.float()[perm]
    q_val = torch.quantile(sample, quantile).item()

    thr = max(error_mult * float(tau), q_val)
    mask = face_error > thr

    if max_faces is not None and int(mask.sum().item()) > max_faces:
        topk = torch.topk(face_error, max_faces).indices
        mask = torch.zeros_like(mask)
        mask[topk] = True
    return mask, thr


def localizeHighErrorFaces(
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    target_points: torch.Tensor,
    target_index: NNIndex,
    face_adjacency: torch.Tensor,
    tau: float,
    error_mult: float = 2.0,
    quantile: float = 0.9,
    min_component_faces: int = 4,
    dilation_rings: int = 1,
    max_faces: Union[int, None] = None,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Full localization: bidirectional error -> threshold -> clean -> dilate.

    Returns:
        region_mask: (F,) bool of faces to refine.
        face_error:  (F,) combined per-face error (for logging / weighting).
        stats: dict with threshold, counts and error summaries.
    """
    fit_v = fitVertexError(deformed_vertices, target_index)
    cov_v = coverageVertexError(deformed_vertices, target_points, device=device)

    fit_f = faceErrorFromVertexError(fit_v, faces)
    cov_f = faceErrorFromVertexError(cov_v, faces)
    face_error = torch.maximum(fit_f, cov_f)

    raw_mask, thr = selectHighErrorFaces(
        face_error, tau, error_mult, quantile, max_faces=max_faces
    )

    # drop tiny noise components
    labels, sizes = connectedFaceComponents(raw_mask, face_adjacency)
    if sizes.size > 0 and min_component_faces > 1:
        small = np.nonzero(sizes < min_component_faces)[0]
        if small.size > 0:
            small_set = np.isin(labels, small)
            cleaned_np = raw_mask.detach().cpu().numpy().copy()
            cleaned_np[small_set] = False
            cleaned = torch.from_numpy(cleaned_np).to(raw_mask.device)
        else:
            cleaned = raw_mask
    else:
        cleaned = raw_mask

    region_mask = dilateFaceMask(cleaned, face_adjacency, dilation_rings)

    stats = {
        "threshold": float(thr),
        "tau": float(tau),
        "n_raw": int(raw_mask.sum().item()),
        "n_region": int(region_mask.sum().item()),
        "face_error_max": float(face_error.max().item()),
        "face_error_mean": float(face_error.mean().item()),
        "n_components": int(sizes.size),
    }
    return region_mask, face_error, stats


def localizePlateauHighErrorFaces(
    deformed_vertices: torch.Tensor,
    faces: torch.Tensor,
    target_points: torch.Tensor,
    target_index: NNIndex,
    face_adjacency: torch.Tensor,
    fit_state: dict,
    level: int,
    tau: float,
    *,
    error_mult: float = 2.0,
    quantile: float = 0.9,
    min_component_faces: int = 4,
    dilation_rings: int = 1,
    max_faces: Optional[int] = None,
    local_drop_tau: float = 0.02,
    max_blocked_vertex_ratio: float = 0.5,
    refine_cooldown: int = 1,
    device: str = "cuda",
):
    """Refine only faces that are (a) high error, (b) LOCALLY plateaued, and
    (c) still optimizable -- the three-condition criterion.

    This replaces the old "global plateau -> refine the current high-error
    faces" rule, which subdivided whatever happened to have high error when the
    GLOBAL residual flattened, including unreachable thin shells whose error
    never falls. Here every condition is local and batched:

      high_error    : combined bidirectional per-face error exceeds the tolerance
                      (the same absolute+quantile test as `selectHighErrorFaces`).
      plateau_face  : the face's residual-drop EMA has flattened (per-face, from
                      `fit_state`), i.e. THIS region stopped improving.
      not blocked   : fewer than `max_blocked_vertex_ratio` of the face's
                      vertices are non-optimizable -- an unreachable shell is
                      skipped so subdivision is never wasted multiplying faces
                      that cannot be pulled onto the target.
      cool-down     : the face was not just created/refined (`last_refine` more
                      than `refine_cooldown` levels old) so a region is not
                      re-split on consecutive rounds.

    Returns (region_mask, face_error, stats).
    """
    fit_v = fitVertexError(deformed_vertices, target_index)
    cov_v = coverageVertexError(deformed_vertices, target_points, device=device)
    fit_f = faceErrorFromVertexError(fit_v, faces)
    cov_f = faceErrorFromVertexError(cov_v, faces)
    face_error = torch.maximum(fit_f, cov_f)

    high_mask, thr = selectHighErrorFaces(
        face_error, tau, error_mult, quantile, max_faces=None
    )

    fstate = faceStateFromVertexState(
        fit_state,
        faces,
        tau,
        local_drop_tau=local_drop_tau,
        max_blocked_vertex_ratio=max_blocked_vertex_ratio,
    )
    cool = (level - fstate["last_refine"]) >= int(refine_cooldown)
    candidate = high_mask & fstate["plateau_face"] & (~fstate["blocked_face"]) & cool

    # cap AFTER the three-condition filter so the budget is spent on faces that
    # are actually worth refining (highest error among the eligible ones).
    if max_faces is not None and int(candidate.sum().item()) > max_faces:
        masked_err = torch.where(
            candidate, face_error, torch.full_like(face_error, -1.0)
        )
        topk = torch.topk(masked_err, max_faces).indices
        capped = torch.zeros_like(candidate)
        capped[topk] = True
        candidate = capped

    n_high = int(high_mask.sum().item())
    n_plateau = int((high_mask & fstate["plateau_face"]).sum().item())
    n_blocked_skipped = int(
        (high_mask & fstate["plateau_face"] & fstate["blocked_face"]).sum().item()
    )
    n_candidate = int(candidate.sum().item())

    # clean tiny noise components, then dilate one ring for a crack-free region
    labels, sizes = connectedFaceComponents(candidate, face_adjacency)
    if sizes.size > 0 and min_component_faces > 1:
        small = np.nonzero(sizes < min_component_faces)[0]
        if small.size > 0:
            small_set = np.isin(labels, small)
            cleaned_np = candidate.detach().cpu().numpy().copy()
            cleaned_np[small_set] = False
            cleaned = torch.from_numpy(cleaned_np).to(candidate.device)
        else:
            cleaned = candidate
    else:
        cleaned = candidate

    region_mask = dilateFaceMask(cleaned, face_adjacency, dilation_rings)

    stats = {
        "threshold": float(thr),
        "tau": float(tau),
        "face_error_max": float(face_error.max().item()),
        "face_error_mean": float(face_error.mean().item()),
        "n_high_error_faces": n_high,
        "n_local_plateau_faces": n_plateau,
        "n_blocked_faces_skipped": n_blocked_skipped,
        "n_candidate_faces": n_candidate,
        "n_region": int(region_mask.sum().item()),
    }
    return region_mask, face_error, stats
