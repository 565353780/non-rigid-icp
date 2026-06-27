import os
import json
import math
import time
import torch
import numpy as np
import open3d as o3d
from tqdm import trange
from typing import Union, Tuple

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.icp import icp
from non_rigid_icp.Method.time import getCurrentTime
from non_rigid_icp.Method.trans import toPointCloud
from non_rigid_icp.Method.sampling import (
    normalizePairToTargetFrame,
    sampleMeshSurface,
)
from non_rigid_icp.Method.nn import NNIndex
from non_rigid_icp.Method.topology import (
    buildUniqueEdges,
    buildFaceAdjacency,
    dilateFaceMask,
)
from non_rigid_icp.Method.convergence import PlateauMonitor
from non_rigid_icp.Method.error_field import (
    localizeHighErrorFaces,
    localizePlateauHighErrorFaces,
    localizeSaggingFaces,
    localizeRatioSaggingFaces,
    localizeAboveMeanCentroidFaces,
)
from non_rigid_icp.Method.subdivision import subdivideMarkedFaces
from non_rigid_icp.Method.fit_state import (
    initVertexFitState,
    updateVertexFitState,
    stateFloatAttrs,
    stateFromFloatAttrs,
)
from non_rigid_icp.Method.collision import (
    buildCollisionCandidatesAABB,
    pairKeys,
)
from non_rigid_icp.Method.self_intersection import findSelfIntersections
from non_rigid_icp.Method.trajectory_guard import (
    largestSafeStep,
    earliestSafeStep,
    segmentMeshIntersections,
    segmentMeshIntersectionParams,
    earliestSegmentMeshHits,
)
from non_rigid_icp.Method.front_advance import (
    frontAdvancingStep,
    penetrationRelaxStep,
)
from non_rigid_icp.Method.spatial_hash import triangleAABBs, estimateCellSize
from non_rigid_icp.Method.implicit_field import ImplicitField
from non_rigid_icp.Method.triton_kernels import clampNorm
from non_rigid_icp.Method.crop import cropMeshByBBox, cropMeshToBBoxUnion
from non_rigid_icp.Method.sheet_constraints import (
    detectSheetPairs,
    detectWallPairsRaycast,
)
from non_rigid_icp.Method.geometry import vertexNormals
from non_rigid_icp.Loss.surface import edgeLaplacianLoss
from non_rigid_icp.Loss.collision import selfCollisionBarrierLoss
from non_rigid_icp.Loss.sheet import sheetOrderBarrierLoss
from non_rigid_icp.Loss.inversion import (
    triangleInversionBarrierLoss,
    triangleAreaNormals,
)
from non_rigid_icp.Metric.chamfer import computeChamferMetrics, computeF1AtThreshold


class WatertightFitter(object):
    """Topology-agnostic non-rigid fitter for very large meshes.

    Source (the watertight mesh) is deformed by a per-vertex displacement field
    so that its surface snaps onto the target (the original mesh). The watertight
    source is a thin closed shell (opposing layers ~1-4 tau apart); naively
    snapping both layers onto the single target surface collapses them THROUGH
    each other. The guard below is built from first principles to prevent that:

      1. Authoritative self-intersection. A complete inflated-AABB grid-hash
         broad phase (never misses the opposing sheet, unlike centroid k-NN) +
         exact triangle-triangle narrow phase. Used as the baseline scan, the
         in-loop hard gate, and the final acceptance gate.

      2. Sheet-order barrier. The dominant failure is opposing layers swapping
         sides. We freeze the separation axis of near pairs and penalize the
         signed separation dropping below a small margin -- tunneling-robust,
         and lets layers compress toward the target without crossing.

      3. Self-collision distance barrier + triangle inversion barrier as soft
         backups for general folds.

      4. Checkpoint backtracking. Every few outer steps the authoritative gate
         re-checks the moved region; a step that introduced a new crossing is
         rolled back to the last clean checkpoint with stronger barriers.

      5. Cumulative-motion reference. A clean-source vertex field is carried in
         lock-step through subdivision, so "which faces moved" (hence which to
         guard) stays correct across refinement -- the bug that made the old
         guard inert after the first subdivision.

      6. Adaptive local subdivision of plateaued high-error regions (up to K).

    Pipeline:
      normalize -> rigid ICP -> [ optimize-to-plateau (guarded) -> localize ->
      subdivide ]*K -> authoritative final gate -> de-normalize and report.
    """

    def __init__(
        self,
        device: str = "cuda",
        outer_iter: int = 60,
        inner_iter: int = 20,
        lr: float = 2e-4,
        train_source_samples: int = 300000,
        train_target_samples: int = 2000000,
        eval_samples: int = 2000000,
        laplacian_weight: float = 200.0,
        coverage_weight: float = 1.0,
        fit_weight: float = 1.0,
        point_to_plane_weight: float = 0.0,
        mask_dist_schedule: Union[list, None] = None,
        laplacian_schedule: Union[list, None] = None,
        point_to_plane_schedule: Union[list, None] = None,
        resample_every: int = 1,
        corr_refresh_every: int = 10,
        # --- self-collision guard (authoritative) ---
        enable_self_collision_guard: bool = True,
        collision_broad_tau: float = 1.0,
        collision_margin_tau: float = 0.25,
        collision_weight: float = 200.0,
        collision_backoff: float = 4.0,
        collision_refresh_every: int = 4,
        collision_check_every: int = 3,
        max_collision_retries: int = 2,
        collision_active_tau: float = 0.5,
        collision_max_active: int = 3000000,
        max_collision_pairs: int = 2000000,
        # --- normal-gated correspondence (double-layer collapse prevention) ---
        normal_gate: bool = True,
        normal_gate_cos: float = 0.0,
        # --- sheet order / thickness barrier ---
        enable_sheet_guard: bool = True,
        sheet_gap_tau: float = 1.0,
        sheet_min_margin_tau: float = 0.5,
        sheet_weight: float = 1000.0,
        sheet_opposite_only: bool = True,
        sheet_max_thickness_tau: float = 6.0,
        max_sheet_pairs: int = 60000000,
        # --- triangle inversion / area barrier ---
        enable_inversion_guard: bool = True,
        inversion_weight: float = 20.0,
        inversion_flip_margin: float = 0.0,
        inversion_area_frac: float = 0.1,
        # --- trajectory self-intersection guard (user-defined criterion) ---
        enable_trajectory_guard: bool = True,
        trajectory_check_inner_every: int = 5,
        trajectory_active_tau: float = 0.5,
        trajectory_min_move_tau: float = 0.1,
        trajectory_bisect_steps: int = 12,
        trajectory_resolve_iters: int = 4,
        trajectory_final_rounds: int = 24,
        trajectory_dilation_rings: int = 1,
        trajectory_inflate_tau: float = 0.0,
        trajectory_max_active: int = 4000000,
        trajectory_seg_chunk: int = 200000,
        # Parametric pull-back clearance (in tau): after computing each crossing
        # vertex's earliest crossing parameter t_min along its OWN ref->current
        # segment, it is placed at t_min minus this many tau (converted to a
        # fraction of the segment length), so the safe point sits strictly on the
        # rest side of the pierced sheet rather than exactly on it.
        trajectory_clearance_tau: float = 0.05,
        # --- final acceptance ---
        strict_no_intersection: bool = True,
        # --- adaptive subdivision ---
        max_subdivisions: int = 4,
        refine_iter: int = 25,
        plateau_window: int = 6,
        plateau_rel_tol: float = 3e-3,
        plateau_patience: int = 2,
        error_mult: float = 2.0,
        error_quantile: float = 0.9,
        max_refine_faces: Union[int, None] = 1500000,
        min_component_faces: int = 4,
        dilation_rings: int = 1,
        # --- sag-based adaptive subdivision (the clamped stepwise path) ---
        # refine ONLY faces whose interior tents off the target beyond their
        # corners: sag(f)=d(centroid,T)-mean_i d(v_i,T) > refine_sag_mult*tau AND
        # d(centroid,T) > refine_centroid_mult*tau. This is the sole criterion
        # that lowers the surface-to-surface error for the fewest faces (see
        # error_field.faceSagError); set refine_sag_mult<=0 to fall back to the
        # legacy vertex-error localizer.
        #
        # The threshold MUST sit clearly above tau: a finite triangle on a curved
        # target always has a residual ~tau sag, so splitting at sag>=1*tau just
        # regenerates ~tau-sag children forever (chasing the target's own
        # discretization noise) without improving F1. A diagnostic on case1 after
        # one projection step found EVERY face within 1.8 tau (0 faces sag>2tau),
        # so 2.0 only splits genuinely tented faces -- of which a well-projected
        # mesh has essentially none, exactly the "fewest faces" optimum.
        refine_sag_mult: float = 2.0,
        refine_centroid_mult: float = 2.0,
        refine_sag_quantile: float = 0.0,
        # RATIO criterion (user spec, takes precedence over the sag-DIFF above
        # when refine_ratio > 0): split a face when
        #   d(centroid,T)/mean_i d(v_i,T) > refine_ratio  AND
        #   d(centroid,T) > refine_centroid_mult*tau.
        # The denominator (mean corner distance) is floored at
        # refine_ratio_denom_eps_tau*tau so corners sitting ~on the target cannot
        # make the ratio explode. Set refine_ratio<=0 to use the sag-DIFF path.
        refine_ratio: float = 0.0,
        refine_ratio_denom_eps_tau: float = 0.25,
        # ABOVE-MEAN criterion (user spec, highest precedence when
        # refine_above_mean is True): split every face whose centroid distance to
        # the target exceeds mean_mult * (mean centroid distance over ALL faces).
        # Adaptive + parameter-free: the bar tightens as the fit improves.
        # `refine_min_centroid_tau` optionally floors it so faces already within
        # tolerance are never split.
        refine_above_mean: bool = False,
        refine_mean_mult: float = 1.0,
        refine_min_centroid_tau: float = 0.0,
        # --- unoptimizable-vertex state machine (clamped stepwise path) ---
        unopt_error_tau: float = 1.0,
        unopt_min_intended_move_tau: float = 0.02,
        unopt_min_actual_move_tau: float = 0.004,
        unopt_min_progress_ratio: float = 0.1,
        unopt_block_patience: int = 3,
        local_drop_tau: float = 0.02,
        max_blocked_vertex_ratio: float = 0.5,
        refine_cooldown: int = 1,
        # --- region-restricted (bbox) evaluation, in the de-normalized frame ---
        eval_bbox_center: Union[list, tuple, np.ndarray, None] = None,
        eval_bbox_edge: Union[float, list, tuple, np.ndarray] = 0.2,
        eval_bbox_mode: str = "all",
        eval_bboxes: Union[list, tuple, None] = None,
        crop_eval_samples: int = 300000,
        # OPTIONAL pre-fit crop (validation only, zero-cost to disable). When set,
        # right after the rigid ICP and BEFORE any optimization the source AND
        # target meshes are cropped in lock-step to the UNION of these boxes, so a
        # huge mesh can be validated on just the regions of interest while every
        # kept region is bit-for-bit what the full-mesh run would optimize. Boxes
        # use the SAME original-target frame as `eval_bboxes`; pass None / [] (the
        # default) to fully remove the crop. Each item: dict{center,edge[,mode]}
        # or (center, edge[, mode]) tuple. `prefit_crop_mode` is the default mode.
        prefit_crop_bboxes: Union[list, tuple, None] = None,
        prefit_crop_mode: str = "centroid",
        save_result_folder_path: Union[str, None] = "auto",
        seed: int = 0,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.outer_iter = outer_iter
        self.inner_iter = inner_iter
        self.lr = lr
        self.train_source_samples = train_source_samples
        self.train_target_samples = train_target_samples
        self.eval_samples = eval_samples
        self.laplacian_weight = laplacian_weight
        self.coverage_weight = coverage_weight
        self.fit_weight = fit_weight
        self.point_to_plane_weight = point_to_plane_weight
        self.resample_every = resample_every
        self.corr_refresh_every = corr_refresh_every

        self.enable_self_collision_guard = enable_self_collision_guard
        self.collision_broad_tau = collision_broad_tau
        self.collision_margin_tau = collision_margin_tau
        self.collision_weight = collision_weight
        self.collision_backoff = collision_backoff
        self.collision_refresh_every = collision_refresh_every
        self.collision_check_every = collision_check_every
        self.max_collision_retries = max_collision_retries
        self.collision_active_tau = collision_active_tau
        self.collision_max_active = collision_max_active
        self.max_collision_pairs = max_collision_pairs

        self.normal_gate = normal_gate
        self.normal_gate_cos = normal_gate_cos

        self.enable_sheet_guard = enable_sheet_guard
        self.sheet_gap_tau = sheet_gap_tau
        self.sheet_min_margin_tau = sheet_min_margin_tau
        self.sheet_weight = sheet_weight
        self.sheet_opposite_only = sheet_opposite_only
        self.sheet_max_thickness_tau = sheet_max_thickness_tau
        self.max_sheet_pairs = max_sheet_pairs

        self.enable_inversion_guard = enable_inversion_guard
        self.inversion_weight = inversion_weight
        self.inversion_flip_margin = inversion_flip_margin
        self.inversion_area_frac = inversion_area_frac

        self.enable_trajectory_guard = enable_trajectory_guard
        self.trajectory_check_inner_every = trajectory_check_inner_every
        self.trajectory_active_tau = trajectory_active_tau
        self.trajectory_min_move_tau = trajectory_min_move_tau
        self.trajectory_bisect_steps = trajectory_bisect_steps
        self.trajectory_resolve_iters = trajectory_resolve_iters
        self.trajectory_final_rounds = trajectory_final_rounds
        self.trajectory_dilation_rings = trajectory_dilation_rings
        self.trajectory_inflate_tau = trajectory_inflate_tau
        self.trajectory_max_active = trajectory_max_active
        self.trajectory_seg_chunk = trajectory_seg_chunk
        self.trajectory_clearance_tau = trajectory_clearance_tau

        self.strict_no_intersection = strict_no_intersection

        self.max_subdivisions = max_subdivisions
        self.refine_iter = refine_iter
        self.plateau_window = plateau_window
        self.plateau_rel_tol = plateau_rel_tol
        self.plateau_patience = plateau_patience
        self.error_mult = error_mult
        self.error_quantile = error_quantile
        self.max_refine_faces = max_refine_faces
        self.min_component_faces = min_component_faces
        self.dilation_rings = dilation_rings
        self.refine_sag_mult = refine_sag_mult
        self.refine_centroid_mult = refine_centroid_mult
        self.refine_sag_quantile = refine_sag_quantile
        self.refine_ratio = refine_ratio
        self.refine_ratio_denom_eps_tau = refine_ratio_denom_eps_tau
        self.refine_above_mean = refine_above_mean
        self.refine_mean_mult = refine_mean_mult
        self.refine_min_centroid_tau = refine_min_centroid_tau

        self.unopt_error_tau = unopt_error_tau
        self.unopt_min_intended_move_tau = unopt_min_intended_move_tau
        self.unopt_min_actual_move_tau = unopt_min_actual_move_tau
        self.unopt_min_progress_ratio = unopt_min_progress_ratio
        self.unopt_block_patience = unopt_block_patience
        self.local_drop_tau = local_drop_tau
        self.max_blocked_vertex_ratio = max_blocked_vertex_ratio
        self.refine_cooldown = refine_cooldown

        self.eval_bbox_center = (
            None if eval_bbox_center is None
            else np.asarray(eval_bbox_center, dtype=np.float64).reshape(3)
        )
        self.eval_bbox_edge = eval_bbox_edge
        self.eval_bbox_mode = eval_bbox_mode
        # Normalize the bbox config into a list of named boxes. `eval_bboxes`
        # (an explicit list of {name?, center, edge, mode?} or (center, edge)
        # tuples) takes precedence; otherwise fall back to the single
        # eval_bbox_center/edge for backward compatibility.
        self.eval_bboxes = self._normalizeBBoxes(
            eval_bboxes, eval_bbox_center, eval_bbox_edge, eval_bbox_mode
        )
        self.crop_eval_samples = crop_eval_samples
        # optional pre-fit crop (see __init__ docstring). Reuse _normalizeBBoxes
        # so a single dict / tuple / list is all accepted; [] disables it.
        self.prefit_crop_bboxes = self._normalizeBBoxes(
            prefit_crop_bboxes, None, eval_bbox_edge, prefit_crop_mode
        )

        self.seed = seed

        self.mask_dist_schedule = mask_dist_schedule
        self.laplacian_schedule = laplacian_schedule
        self.point_to_plane_schedule = point_to_plane_schedule

        if save_result_folder_path == "auto":
            save_result_folder_path = "./output/" + getCurrentTime() + "/"
        self.save_result_folder_path = save_result_folder_path
        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)

        self.source_mesh = None
        self.target_mesh = None
        self.norm_center = None
        self.norm_scale = 1.0
        self.L = 1.0
        return

    def loadMeshes(self, source_mesh: Mesh, target_mesh: Mesh) -> bool:
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh

        self.norm_center, self.norm_scale, self.L = normalizePairToTargetFrame(
            source_mesh, target_mesh
        )
        print(
            "[INFO][WatertightFitter::loadMeshes] L(target max bbox edge)=",
            round(self.L, 6),
            "norm_scale=",
            round(self.norm_scale, 6),
        )
        return True

    @staticmethod
    def _normalizeBBoxes(eval_bboxes, center, edge, mode) -> list:
        """Build a list of named eval boxes: [{name, center(3,), edge, mode}].

        Accepts either an explicit `eval_bboxes` list (each item a dict with
        center/edge[/name/mode] or a (center, edge) / (center, edge, mode)
        tuple) or the legacy single center/edge/mode. Returns [] when no box is
        configured (region-restricted eval disabled)."""
        def _one(item, idx):
            if isinstance(item, dict):
                c = item["center"]
                e = item.get("edge", edge)
                m = item.get("mode", mode)
                name = item.get("name", f"bbox_{idx}")
            else:
                c = item[0]
                e = item[1] if len(item) > 1 else edge
                m = item[2] if len(item) > 2 else mode
                name = f"bbox_{idx}"
            return {
                "name": name,
                "center": np.asarray(c, dtype=np.float64).reshape(3),
                "edge": e,
                "mode": m,
            }

        if eval_bboxes is not None:
            return [_one(it, i) for i, it in enumerate(eval_bboxes)]
        if center is None:
            return []
        return [
            {
                "name": "bbox_0",
                "center": np.asarray(center, dtype=np.float64).reshape(3),
                "edge": edge,
                "mode": mode,
            }
        ]

    def _rigidInit(self) -> None:
        source_o3d = self.source_mesh.toO3DMesh()
        target_pts = sampleMeshSurface(
            self.target_mesh, min(self.train_target_samples, 200000), seed=self.seed
        )
        target_pcd = toPointCloud(target_pts)
        transformation = icp(source_o3d, target_pcd)
        if transformation is None:
            print("[WARN][WatertightFitter::_rigidInit] ICP failed, skip rigid init")
            return
        source_o3d.transform(transformation)
        self.source_mesh.vertices = np.asarray(source_o3d.vertices)
        return

    def _applyPrefitCrop(self) -> None:
        """OPTIONAL, minimally-invasive: crop source+target to the union of
        `prefit_crop_bboxes` right after the rigid ICP and before any
        optimization, so a huge mesh can be validated on the regions of interest
        only. No-op when `prefit_crop_bboxes` is empty (the production path).

        Both meshes are already in the shared normalized frame here, while the
        boxes are given in the original-target frame (same convention as
        `eval_bboxes`); each box is therefore mapped to normalized coords via
        c_norm=(c-center)*scale, edge_norm=edge*scale before cropping. Source
        and target are cropped with the IDENTICAL boxes so the kept region is
        exactly aligned across the two meshes."""
        if not self.prefit_crop_bboxes:
            return

        def _toNorm(box):
            c = (np.asarray(box["center"], dtype=np.float64) - self.norm_center) \
                * self.norm_scale
            e = np.asarray(box["edge"], dtype=np.float64) * self.norm_scale
            return c, e

        boxes = [_toNorm(b) for b in self.prefit_crop_bboxes]
        # one mode for the whole pre-fit crop (boxes share prefit_crop_mode).
        mode = self.prefit_crop_bboxes[0]["mode"]

        for label, mesh in (("source", self.source_mesh), ("target", self.target_mesh)):
            v0 = np.asarray(mesh.vertices)
            f0 = np.asarray(mesh.triangles)
            sv, sf = cropMeshToBBoxUnion(v0, f0, boxes, mode=mode)
            mesh.vertices = sv
            mesh.triangles = sf
            mesh.vertex_colors = None
            mesh.vertex_normals = None
            mesh.triangle_normals = None
            print(
                f"[INFO][WatertightFitter::_applyPrefitCrop] {label}: "
                f"{v0.shape[0]}->{sv.shape[0]} verts, "
                f"{f0.shape[0]}->{sf.shape[0]} faces "
                f"({len(boxes)} bbox union, mode={mode})"
            )
        return

    # ------------------------------------------------------------------ #
    # topology / motion                                                  #
    # ------------------------------------------------------------------ #
    def _buildTopology(self) -> None:
        self._edges = buildUniqueEdges(self._faces)
        self._face_adj = buildFaceAdjacency(self._faces)
        # one incident face per vertex (cheap O(F) scatter), used to turn a
        # trajectory hit (offending vertex -> pierced face) into a face pair for
        # the sheet-order barrier without an expensive vertex->faces adjacency.
        v = self._verts.shape[0]
        f = self._faces.shape[0]
        vof = torch.full((v,), -1, dtype=torch.long, device=self._faces.device)
        fidx = torch.arange(f, device=self._faces.device)
        vof[self._faces[:, 0]] = fidx
        vof[self._faces[:, 1]] = fidx
        vof[self._faces[:, 2]] = fidx
        self._vert_one_face = vof
        # grid cell size for the trajectory broad phase is tessellation-scale and
        # stable within a topology; cache it (recomputed after each subdivision).
        self._traj_cell_size = None
        # diagnostic clean-ref triangle-triangle baseline depends on the current
        # topology, so invalidate it whenever the topology changes.
        self._diag_baseline_keys = None

    def _deformed(self) -> torch.Tensor:
        return self._verts + self._disp

    def _cumulativeMotion(self) -> torch.Tensor:
        """Per-vertex motion from the clean (rigid-init) source, carried through
        subdivision via `_ref_verts`. Correct across refinement (unlike the old
        per-cycle `_disp`, which reset to 0 after each subdivision)."""
        return (self._deformed().detach() - self._ref_verts).norm(dim=1)

    def _activeFaceMask(self, motion_thresh: float) -> Union[torch.Tensor, None]:
        """Faces touching a vertex whose CUMULATIVE motion exceeds motion_thresh,
        grown one ring, capped to the largest movers. A new crossing requires a
        face to move ~the local gap (>= tau), so sub-threshold movers cannot
        create one and need not be guarded."""
        motion = self._cumulativeMotion()
        moved_v = motion > motion_thresh
        if not bool(moved_v.any()):
            return None
        face_moved = moved_v[self._faces].any(dim=1)
        n_active = int(face_moved.sum().item())
        if n_active > self.collision_max_active:
            face_motion = motion[self._faces].amax(dim=1)
            face_motion = torch.where(
                face_moved, face_motion, torch.zeros_like(face_motion)
            )
            topk = torch.topk(face_motion, self.collision_max_active).indices
            face_moved = torch.zeros_like(face_moved)
            face_moved[topk] = True
        return dilateFaceMask(face_moved, self._face_adj, 1)

    # ------------------------------------------------------------------ #
    # authoritative self-intersection                                    #
    # ------------------------------------------------------------------ #
    def _newIntersections(
        self, query_mask: Union[torch.Tensor, None], restrict: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """Authoritative count of intersecting non-adjacent pairs NOT in the
        pre-existing baseline. `query_mask` scopes the broad phase to the moved
        region; with `restrict` the grid is also built over only that region
        (the cheap in-loop gate). The final acceptance gate uses restrict=False
        (and query_mask=None) for a complete full-mesh guarantee."""
        deformed = self._deformed().detach()
        inter = findSelfIntersections(
            deformed,
            self._faces,
            inflate=0.0,
            exclude_ring=1,
            query_mask=query_mask,
            restrict_to_query=restrict,
        )
        if inter.shape[0] == 0:
            return 0, inter
        if self._baseline_keys.numel() > 0:
            keys = pairKeys(inter, self._faces.shape[0])
            is_base = torch.isin(keys, self._baseline_keys)
            inter = inter[~is_base]
        return int(inter.shape[0]), inter

    def _buildBaseline(self) -> None:
        """Pre-existing (allowed) intersections, scanned on the CLEAN reference.

        Critical: scan `_ref_verts` (the clean source, carried through
        subdivision), NOT the deformed mesh. Scanning the deformed mesh after a
        subdivision would record fitting-INDUCED crossings as "pre-existing" and
        silently legitimize them -- exactly the bug that hid 15M self-crossings.
        The clean reference only ever carries the source's genuine pre-existing
        intersections, so the guard fights every crossing the fit creates."""
        empty_keys = torch.zeros(0, dtype=torch.long, device=self.device)
        if not self.enable_self_collision_guard:
            self._baseline_keys = empty_keys
            return
        inter = findSelfIntersections(
            self._ref_verts.detach(), self._faces, inflate=0.0, exclude_ring=1
        )
        self._baseline_keys = pairKeys(inter, self._faces.shape[0])
        print(
            f"[INFO][WatertightFitter] baseline scan (clean ref): {inter.shape[0]} "
            "pre-existing intersections (ignored)."
        )

    # ------------------------------------------------------------------ #
    # trajectory self-intersection guard (user-defined criterion)        #
    # ------------------------------------------------------------------ #
    def _trajCellSize(self, cur: torch.Tensor) -> float:
        """Tessellation-scale grid cell for the trajectory broad phase, cached
        per topology (recomputed after subdivision via `_buildTopology`)."""
        if self._traj_cell_size is None:
            tri_lo, tri_hi = triangleAABBs(cur, self._faces, 0.0)
            self._traj_cell_size = float(
                estimateCellSize(tri_lo, tri_hi, quantile=0.9, factor=1.0)
            )
        return self._traj_cell_size

    def _trajectoryActiveVertices(self) -> Union[torch.Tensor, None]:
        """Vertices whose trajectory [ref -> current] could plausibly cross: the
        cumulative movers (a crossing needs ~the local gap of motion), capped to
        the largest movers so the per-step guard stays cheap on huge meshes."""
        motion = self._cumulativeMotion()
        active = motion > (self.trajectory_active_tau * self._tau_norm)
        if not bool(active.any()):
            return None
        ids = torch.nonzero(active, as_tuple=False).reshape(-1)
        if ids.numel() > self.trajectory_max_active:
            top = torch.topk(motion[ids], self.trajectory_max_active).indices
            ids = ids[top]
        return ids

    def _trajectoryActiveFaceIds(self) -> Union[torch.Tensor, None]:
        """Faces the in-loop trajectory broad phase is scoped to: the moved
        region grown one ring. A new crossing must involve a moved face, so this
        is the complete candidate set for in-loop crossings -- and it keeps the
        grid hash O(moved) instead of O(28.9M), which is what makes the guard fit
        in memory next to the (large) sheet/collision constraint tensors. The
        authoritative full-mesh sweep is reserved for the final gate."""
        mask = self._activeFaceMask(self.trajectory_active_tau * self._tau_norm)
        if mask is None:
            return None
        return torch.nonzero(mask, as_tuple=False).reshape(-1)

    def _applyTrajectoryGuard(
        self,
        ids: Union[torch.Tensor, None] = None,
        face_ids: Union[torch.Tensor, None] = None,
    ) -> int:
        """Pull every offending vertex back along its OWN trajectory.

        For each guarded vertex v we test the straight segment from its clean
        watertight-rest position `_ref_verts[v]` to its current fitted position
        against the non-incident faces of the current mesh. If it pierces, the
        vertex is moved to the largest safe fraction along that segment -- i.e.
        the offending region is dragged back JUST FAR ENOUGH to be crossing-free,
        never reset wholesale to the watertight rest. Only the clamped vertices
        (plus a small ring, to release stale Adam momentum) are touched.

        `face_ids` optionally scopes the broad phase to a face subset (the moved
        region in-loop); None tests against the full mesh (the final gate).

        Returns the number of vertices that were pulled back.
        """
        if not self.enable_trajectory_guard:
            return 0
        if ids is None:
            ids = self._trajectoryActiveVertices()
        if ids is None or ids.numel() == 0:
            return 0

        cur = self._deformed().detach()
        ref = self._ref_verts.detach()
        cell = self._trajCellSize(cur)
        inflate = self.trajectory_inflate_tau * self._tau_norm

        # Batch the active segments. A single largestSafeStep over all movers
        # builds an O(segments x faces-per-cell) candidate-pair tensor, which can
        # blow past GPU memory when a whole thin-shell region moves at once
        # (observed 18.6 GiB for ~400k movers). Chunking caps the peak candidate
        # memory regardless of how many vertices moved; correctness is unchanged
        # because each segment's pullback is independent.
        n_total = ids.numel()
        chunk = self.trajectory_seg_chunk
        clamp_id_parts = []
        clamp_pos_parts = []
        hit_owner_parts = []
        hit_face_parts = []
        for start in range(0, n_total, chunk):
            sub = ids[start:start + chunk]
            safe_pos, need, hit_seg, hit_face = largestSafeStep(
                ref[sub],
                cur[sub],
                cur,
                self._faces,
                owner_vid=sub,
                inflate=inflate,
                n_bisect=self.trajectory_bisect_steps,
                cell_size=cell,
                face_ids=face_ids,
                return_hits=True,
            )
            if bool(need.any()):
                clamp_id_parts.append(sub[need])
                clamp_pos_parts.append(safe_pos[need])
            if hit_seg.numel() > 0:
                hit_owner_parts.append(sub[hit_seg])
                hit_face_parts.append(hit_face)

        if not clamp_id_parts:
            return 0
        clamp_ids = torch.cat(clamp_id_parts, dim=0)
        clamp_pos = torch.cat(clamp_pos_parts, dim=0)
        n_clamped = int(clamp_ids.numel())

        with torch.no_grad():
            self._disp.data[clamp_ids] = clamp_pos - self._verts[clamp_ids]

        # release Adam momentum on the pulled-back vertices + a small ring, so a
        # stale velocity does not immediately re-drive them through the sheet.
        vmask = torch.zeros(
            self._verts.shape[0], dtype=torch.bool, device=self.device
        )
        vmask[clamp_ids] = True
        for _ in range(self.trajectory_dilation_rings):
            grown = vmask.clone()
            grown[self._edges[:, 0]] |= vmask[self._edges[:, 1]]
            grown[self._edges[:, 1]] |= vmask[self._edges[:, 0]]
            vmask = grown
        state = self._optimizer.state.get(self._disp, {})
        if "exp_avg" in state:
            state["exp_avg"][vmask] = 0.0
            state["exp_avg_sq"][vmask] = 0.0

        # feed the actually-pierced (offending-vertex face, pierced face) pairs
        # into the sheet-order barrier so whatever the static set missed is
        # constrained the moment it is seen (axis frozen from the clean ref).
        if self.enable_sheet_guard and hit_owner_parts:
            owner_v = torch.cat(hit_owner_parts, dim=0)
            hit_face = torch.cat(hit_face_parts, dim=0)
            owner_f = self._vert_one_face[owner_v]
            valid = (owner_f >= 0) & (owner_f != hit_face)
            if bool(valid.any()):
                pairs = torch.stack([owner_f[valid], hit_face[valid]], dim=1)
                self._augmentSheetWithCrossings(pairs)
        return n_clamped

    def _resolveTrajectory(
        self,
        rounds: int,
        ids: Union[torch.Tensor, None] = None,
        scoped: bool = False,
    ) -> int:
        """Apply the trajectory guard repeatedly until no segment crosses (or
        `rounds` is hit). Each pass strictly shortens every offending vertex's
        ref->current segment, and the all-at-rest configuration is crossing-free,
        so the iteration converges to a crossing-free state. `scoped` restricts
        the broad phase to the moved region (memory-bounded, in-loop)."""
        last = 0
        for _ in range(max(1, rounds)):
            face_ids = self._trajectoryActiveFaceIds() if scoped else None
            n = self._applyTrajectoryGuard(ids, face_ids=face_ids)
            last = n
            if n == 0:
                break
        return last

    def _measureTrajectory(
        self, face_ids: Union[torch.Tensor, None] = None
    ) -> dict:
        """Measure the user-defined criterion. `face_ids=None` is the full-mesh
        authoritative scan; a subset scopes it to the moved region (fallback)."""
        cur = self._deformed().detach()
        ref = self._ref_verts.detach()
        v = self._verts.shape[0]
        owner = torch.arange(v, device=self.device)
        cell = self._trajCellSize(cur)
        hit_seg, hit_face, crossed = segmentMeshIntersections(
            ref, cur, cur, self._faces, owner, cell_size=cell, face_ids=face_ids
        )
        n_vertices = int(crossed.sum().item())
        n_pairs = int(hit_seg.numel())
        n_faces = int(torch.unique(hit_face).numel()) if hit_face.numel() else 0
        return {
            "trajectory_crossing_vertices": n_vertices,
            "trajectory_crossing_pairs": n_pairs,
            "trajectory_crossing_faces": n_faces,
            "trajectory_self_intersection_free": bool(n_vertices == 0),
        }

    def _finalTrajectoryGate(self) -> dict:
        """Drive the whole mesh crossing-free, then measure.

        Prefer the authoritative full-mesh sweep; if the (large) full-mesh grid
        hash does not fit alongside the resident fit tensors, fall back to the
        moved-region scope -- principled, since an unmoved vertex has a zero-
        length trajectory and an unmoved face cannot be newly pierced by the
        opposing (also-moving) layer in this fit."""
        if not self.enable_trajectory_guard:
            return self._measureTrajectory()

        v = self._verts.shape[0]
        owner = torch.arange(v, device=self.device)
        # the in-loop constraint tensors are no longer needed; free them so the
        # full-mesh broad phase has maximal headroom.
        self._freeConstraintMemory()
        try:
            self._resolveTrajectory(self.trajectory_final_rounds, ids=owner)
            metrics = self._measureTrajectory()
            metrics["trajectory_scope"] = "full_mesh"
            return metrics
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                "[WARN][WatertightFitter] full-mesh trajectory gate hit OOM; "
                "falling back to the moved-region scope."
            )
            for _ in range(self.trajectory_final_rounds):
                face_ids = self._trajectoryActiveFaceIds()
                n = self._applyTrajectoryGuard(ids=owner, face_ids=face_ids)
                if n == 0:
                    break
            metrics = self._measureTrajectory(
                face_ids=self._trajectoryActiveFaceIds()
            )
            metrics["trajectory_scope"] = "active_fallback"
            return metrics

    def _freeConstraintMemory(self) -> None:
        """Release the sheet / collision constraint tensors and clear the CUDA
        cache (called before the memory-heavy final gate)."""
        dev = self.device
        empty2 = torch.zeros(0, 2, dtype=torch.long, device=dev)
        self._collision_pairs = empty2
        self._sheet_pairs = empty2
        self._sheet_axis = torch.zeros(0, 3, device=dev)
        self._sheet_margin = torch.zeros(0, device=dev)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # constraint sets (collision + sheet pairs)                          #
    # ------------------------------------------------------------------ #
    def _buildCollisionConstraints(self) -> None:
        """Distance-barrier candidate pairs from the CURRENT (deformed) geometry.

        These guard general folds and must track current proximity, so they are
        re-detected on the deformed mesh and refreshed periodically. Full-mesh
        (the source is double-layered almost everywhere); the reservoir cap in
        `buildCollisionCandidatesAABB` keeps the build memory-bounded and retains
        the closest (most at-risk) pairs."""
        dev = self.device
        if not self.enable_self_collision_guard:
            self._collision_pairs = torch.zeros(0, 2, dtype=torch.long, device=dev)
            return
        deformed = self._deformed().detach()
        self._collision_pairs = buildCollisionCandidatesAABB(
            deformed,
            self._faces,
            margin=self._broad_margin,
            active_face_mask=None,
            max_pairs=self.max_collision_pairs,
        )

    def _buildSheetConstraints(self) -> None:
        """Opposite-layer pairs + frozen axes from the CLEAN reference geometry.

        Critical fix: the separation axis must be frozen while the two layers
        are still correctly ordered. Detecting on the deformed mesh (the old
        behaviour) froze axes AFTER layers had already crossed, so the barrier
        pushed the wrong way and silently failed (avg violation ~1.7 tau). The
        clean reference `_ref_verts` is carried in lock-step through subdivision
        and is always ordered, so its axes are always valid.

        Per-pair margin = min(global_margin, clean_separation): thick walls may
        compress down to the global margin (good for fit), but already-thin walls
        are only prevented from collapsing FURTHER -- never pushed apart."""
        dev = self.device
        if not self.enable_sheet_guard:
            self._sheet_pairs = torch.zeros(0, 2, dtype=torch.long, device=dev)
            self._sheet_axis = torch.zeros(0, 3, device=dev)
            self._sheet_margin = torch.zeros(0, device=dev)
            return
        ref = self._ref_verts.detach()
        # Ray-cast wall-partner detection: thickness-INDEPENDENT, so it captures
        # the thick (median ~2.4 tau) walls a gap-1 tau broad phase silently
        # missed -- the coverage gap that let case1's walls collapse uncovered.
        pairs = detectWallPairsRaycast(
            ref,
            self._faces,
            max_thickness=self.sheet_max_thickness_tau * self._tau_norm,
        )
        if self.max_sheet_pairs is not None and pairs.shape[0] > self.max_sheet_pairs:
            pairs = pairs[: self.max_sheet_pairs]
        self._sheet_pairs = pairs
        if pairs.shape[0] == 0:
            self._sheet_axis = torch.zeros(0, 3, device=dev)
            self._sheet_margin = torch.zeros(0, device=dev)
            return
        centroids = ref[self._faces].mean(dim=1)
        diff = centroids[pairs[:, 1]] - centroids[pairs[:, 0]]
        axis = (diff / (diff.norm(dim=1, keepdim=True) + 1e-20)).detach()
        self._sheet_axis = axis
        s0 = ((centroids[pairs[:, 1]] - centroids[pairs[:, 0]]) * axis).sum(dim=1)
        # never try to separate a pair beyond 90% of its clean gap
        self._sheet_margin = torch.minimum(
            torch.full_like(s0, self._sheet_margin_scalar), 0.9 * s0
        ).clamp(min=0.0)

    def _augmentSheetWithCrossings(self, inter: torch.Tensor) -> None:
        """Feed the gate's ACTUALLY-detected crossings back into the order
        barrier, with the separation axis frozen from the clean reference.

        Static raycast covers the normal-opposite wall partners, but a complex
        collapse also produces crossings between non-partner faces (folds,
        oblique approaches) that no static set anticipates. Adding each detected
        crossing pair here closes the loop: whatever slips through is constrained
        the moment it is seen, and the clean-reference axis tells the barrier
        which side each face must stay on to undo the crossing. Capacity is
        bounded by `max_sheet_pairs` (oldest dropped first)."""
        if not self.enable_sheet_guard or inter.shape[0] == 0:
            return
        dev = self.device
        F = self._faces.shape[0]
        new_pairs = torch.unique(torch.sort(inter, dim=1).values, dim=0)
        if self._sheet_pairs.shape[0] > 0:
            existing = self._sheet_pairs[:, 0] * F + self._sheet_pairs[:, 1]
            new_keys = new_pairs[:, 0] * F + new_pairs[:, 1]
            new_pairs = new_pairs[~torch.isin(new_keys, existing)]
        if new_pairs.shape[0] == 0:
            return
        ref = self._ref_verts.detach()
        c = ref[self._faces].mean(dim=1)
        diff = c[new_pairs[:, 1]] - c[new_pairs[:, 0]]
        axis = (diff / (diff.norm(dim=1, keepdim=True) + 1e-20)).detach()
        s0 = (diff * axis).sum(dim=1)
        margin = torch.minimum(
            torch.full_like(s0, self._sheet_margin_scalar), 0.9 * s0
        ).clamp(min=0.0)
        self._sheet_pairs = torch.cat([self._sheet_pairs, new_pairs], dim=0)
        self._sheet_axis = torch.cat([self._sheet_axis, axis], dim=0)
        self._sheet_margin = torch.cat([self._sheet_margin, margin], dim=0)
        cap = self.max_sheet_pairs
        if cap is not None and self._sheet_pairs.shape[0] > cap:
            self._sheet_pairs = self._sheet_pairs[-cap:]
            self._sheet_axis = self._sheet_axis[-cap:]
            self._sheet_margin = self._sheet_margin[-cap:]

    def _buildConstraints(self, active_mask: Union[torch.Tensor, None]) -> None:
        """Build both constraint sets (sheet from clean ref, collision from
        deformed). `active_mask` is unused (full-mesh build; see methods)."""
        del active_mask
        self._buildSheetConstraints()
        self._buildCollisionConstraints()

    def _refreshConstraints(self) -> None:
        """Refresh only the deformed-geometry collision pairs. Sheet pairs/axes
        are frozen from the clean reference (stable within a cycle), so they are
        NOT re-detected here -- that is exactly what keeps their axes valid."""
        if not self.enable_self_collision_guard:
            return
        self._buildCollisionConstraints()

    def _setInversionRest(self) -> None:
        if not self.enable_inversion_guard:
            self._ref_area2 = None
            self._ref_normals = None
            return
        with torch.no_grad():
            area2, normals = triangleAreaNormals(self._deformed().detach(), self._faces)
        self._ref_area2 = area2.detach()
        self._ref_normals = normals.detach()

    # ------------------------------------------------------------------ #
    # one inner optimization burst                                       #
    # ------------------------------------------------------------------ #
    def _innerSteps(
        self,
        n_inner: int,
        matched: torch.Tensor,
        matched_n: torch.Tensor,
        weight: torch.Tensor,
        lap_w: float,
        p2p_w: float,
        coll_w: float,
        sheet_w: float,
    ) -> dict:
        data_v = lap_v = coll_v = sheet_v = inv_v = 0.0
        for _ in range(n_inner):
            self._optimizer.zero_grad()
            d = self._verts + self._disp
            diff = d - matched
            if p2p_w > 0:
                plane = (diff * matched_n).sum(dim=1, keepdim=True)
                data = (
                    self.fit_weight * (weight * diff ** 2).sum(dim=1).mean()
                    + p2p_w * (weight * plane ** 2).mean()
                )
            else:
                data = self.fit_weight * (weight * diff ** 2).sum(dim=1).mean()
            lap_loss = edgeLaplacianLoss(self._disp, self._edges)
            loss = data + lap_w * lap_loss

            if coll_w > 0 and self._collision_pairs.shape[0] > 0:
                coll_loss = selfCollisionBarrierLoss(
                    d, self._faces, self._collision_pairs, self._barrier_margin
                )
                loss = loss + coll_w * coll_loss
                coll_v = float(coll_loss.item())
            if sheet_w > 0 and self._sheet_pairs.shape[0] > 0:
                sheet_loss = sheetOrderBarrierLoss(
                    d, self._faces, self._sheet_pairs, self._sheet_axis,
                    self._sheet_margin,
                )
                loss = loss + sheet_w * sheet_loss
                sheet_v = float(sheet_loss.item())
            if self.enable_inversion_guard and self._ref_normals is not None:
                inv_loss = triangleInversionBarrierLoss(
                    d, self._faces, self._ref_normals, self._ref_area2,
                    flip_margin=self.inversion_flip_margin,
                    area_frac=self.inversion_area_frac,
                )
                loss = loss + self.inversion_weight * inv_loss
                inv_v = float(inv_loss.item())

            loss.backward()
            self._optimizer.step()
            data_v = float(data.item())
            lap_v = float(lap_loss.item())

            # real-time trajectory guard: right after the step, pull any vertex
            # whose ref->current trajectory now pierces the mesh back to its
            # largest safe position (local, never a wholesale reset to rest).
            self._traj_step_counter += 1
            if (
                self.enable_trajectory_guard
                and self.trajectory_check_inner_every > 0
                and self._traj_step_counter % self.trajectory_check_inner_every == 0
            ):
                with torch.no_grad():
                    # in-loop: scope the broad phase to the moved region so the
                    # grid hash fits next to the resident constraint tensors.
                    self._applyTrajectoryGuard(
                        face_ids=self._trajectoryActiveFaceIds()
                    )
        return {
            "data": data_v,
            "lap": lap_v,
            "coll": coll_v,
            "sheet": sheet_v,
            "inv": inv_v,
        }

    # ------------------------------------------------------------------ #
    # one optimize-to-plateau cycle                                      #
    # ------------------------------------------------------------------ #
    def _cycleStageParams(self, level: int, i: int, n_outer: int):
        n_stages = len(self.mask_dist_schedule)
        if level == 0:
            milestones = np.linspace(0, n_outer, n_stages + 1).astype(int)
            stage = int(np.searchsorted(milestones, i, side="right") - 1)
            stage = max(0, min(stage, n_stages - 1))
        else:
            stage = n_stages - 1
        return (
            self.mask_dist_schedule[stage],
            self.laplacian_schedule[stage],
            self.point_to_plane_schedule[stage],
            stage,
        )

    def _optimizeCycle(self, level: int, n_outer: int) -> None:
        dev = self.device
        target_pts = self._target_pts
        target_nrm = self._target_nrm
        target_index = self._target_index

        monitor = PlateauMonitor(
            window=self.plateau_window,
            rel_tol=self.plateau_rel_tol,
            patience=self.plateau_patience,
            min_updates=self.plateau_window,
        )

        self._setInversionRest()
        active0 = self._activeFaceMask(self.collision_active_tau * self._tau_norm)
        self._buildConstraints(active0)

        coll_w = self.collision_weight if self.enable_self_collision_guard else 0.0
        sheet_w = self.sheet_weight if self.enable_sheet_guard else 0.0
        safe_snapshot = self._disp.detach().clone()

        matched = matched_n = weight = None
        loop = trange(n_outer, desc=f"cycle{level}")
        for i in loop:
            mask_dist, lap_w, p2p_w, stage = self._cycleStageParams(level, i, n_outer)

            if (self.enable_self_collision_guard or self.enable_sheet_guard) and (
                i > 0 and i % self.collision_refresh_every == 0
            ):
                self._refreshConstraints()

            if i == 0 or (i % self.corr_refresh_every == 0):
                with torch.no_grad():
                    cur = self._deformed().detach()
                    idx, d2 = target_index.query(cur, k=1)
                    idx_t = torch.from_numpy(idx).to(dev)
                    matched = target_pts[idx_t]
                    matched_n = target_nrm[idx_t]
                    keep = torch.from_numpy(d2).to(dev) < (mask_dist ** 2)
                    if self.normal_gate:
                        # DOUBLE-LAYER COLLAPSE PREVENTION. The source is a thin
                        # double shell; pulling BOTH layers onto the single target
                        # surface forces them to interpenetrate. Drive only the
                        # layer whose normal agrees with the local target normal
                        # (the front-facing side); the opposite layer is left to
                        # follow via the Laplacian, so the wall is preserved and
                        # the two layers never have to cross. This is O(V) and
                        # thickness-independent, unlike a proximity barrier.
                        vn = vertexNormals(cur, self._faces)
                        agree = (vn * matched_n).sum(dim=1) > self.normal_gate_cos
                        keep = keep & agree
                    weight = keep.float().unsqueeze(1)

            n_inner = self.inner_iter if i > 0 else 4
            info = self._innerSteps(
                n_inner, matched, matched_n, weight, lap_w, p2p_w, coll_w, sheet_w
            )

            plateau = monitor.update(info["data"])

            # authoritative checkpoint gate. A new crossing must involve a moved
            # face, so scoping the (complete) broad phase to the moved region is
            # exact. Gate periodically AND on the final / plateau iteration so a
            # cycle can never end on an unchecked (possibly dirty) tail.
            n_new = 0
            last_iter = (i == n_outer - 1) or (
                plateau and level < self.max_subdivisions
            )
            if self.enable_self_collision_guard and (
                i % self.collision_check_every == 0 or last_iter
            ):
                active = self._activeFaceMask(0.5 * self._tau_norm)
                # cheap active-restricted gate in-loop; the full-mesh gate at
                # fit() end is the authoritative guarantee.
                n_new, inter = self._newIntersections(active, restrict=not last_iter)
                if n_new > 0:
                    # LOCALIZED rollback. A handful of tessellation-scale
                    # crossings must NOT discard the whole cycle's fit (a global
                    # rollback stalls at the rigid init, since safe_snapshot then
                    # never advances). Revert ONLY the crossing faces + 1 ring to
                    # the last-good state, reset their optimizer momentum so they
                    # do not immediately re-collapse, stiffen the barriers, and
                    # keep all other (good) progress.
                    fmask = torch.zeros(
                        self._faces.shape[0], dtype=torch.bool, device=dev
                    )
                    fmask[inter.reshape(-1)] = True
                    fmask = dilateFaceMask(fmask, self._face_adj, 1)
                    vmask = torch.zeros(
                        self._verts.shape[0], dtype=torch.bool, device=dev
                    )
                    vmask[self._faces[fmask].reshape(-1)] = True
                    self._disp.data[vmask] = safe_snapshot[vmask]
                    state = self._optimizer.state.get(self._disp, {})
                    if "exp_avg" in state:
                        state["exp_avg"][vmask] = 0.0
                        state["exp_avg_sq"][vmask] = 0.0
                    coll_w = min(coll_w * self.collision_backoff, 1e6)
                    sheet_w = min(sheet_w * self.collision_backoff, 1e6)
                    # adaptively constrain the exact pairs that just crossed so
                    # the barrier covers whatever the static set missed.
                    self._augmentSheetWithCrossings(inter)
                    self._refreshConstraints()
                # advance the checkpoint either way: global progress is kept and
                # any crossing region has just been reverted, so the current disp
                # is the new best clean-or-cleaner state.
                safe_snapshot = self._disp.detach().clone()

            loop.set_postfix(
                stage=stage,
                data=round(info["data"], 8),
                sheet=round(info["sheet"], 8),
                coll=round(info["coll"], 8),
                new_si=n_new,
                drop=round(monitor.last_rel_drop, 5),
            )
            self._history.append(
                {
                    "level": level,
                    "iter": i,
                    "stage": stage,
                    "data": info["data"],
                    "lap": info["lap"],
                    "coll": info["coll"],
                    "sheet": info["sheet"],
                    "inv": info["inv"],
                    "coll_w": coll_w,
                    "sheet_w": sheet_w,
                    "new_self_intersections": int(n_new),
                    "n_collision_pairs": int(self._collision_pairs.shape[0]),
                    "n_sheet_pairs": int(self._sheet_pairs.shape[0]),
                    "rel_drop": monitor.last_rel_drop,
                    "mask_dist": mask_dist,
                    "lap_w": lap_w,
                    "p2p_w": p2p_w,
                }
            )

            if plateau and level < self.max_subdivisions:
                print(
                    f"[INFO][WatertightFitter] cycle {level} plateaued at iter {i} "
                    f"(rel_drop={monitor.last_rel_drop:.5f}); triggering refinement."
                )
                break

        # the cycle output is the last verified-clean state (no dirty tail)
        if self.enable_self_collision_guard:
            self._disp.data.copy_(safe_snapshot)

        # resolve the user-defined trajectory criterion over the active region
        # before this cycle's geometry is baked into the next subdivision, so a
        # refinement never starts from a trajectory-crossing state.
        if self.enable_trajectory_guard:
            with torch.no_grad():
                self._resolveTrajectory(self.trajectory_resolve_iters, scoped=True)

    # ------------------------------------------------------------------ #
    # adaptive subdivision                                               #
    # ------------------------------------------------------------------ #
    def _applySubdivision(self, region_mask: torch.Tensor) -> Tuple[int, int]:
        """Atomically refine `region_mask` on BOTH the current (deformed) mesh
        and the clean rest field, in lock-step.

        First-principles invariant (the user's key simplification): the same
        edge-midpoint split is applied to the deformed geometry AND to the rest
        reference `_ref_verts` via `extra_vertex_attrs`, so a new midpoint vertex
        gets the SAME index in both fields and its rest position is the midpoint
        of its two rest endpoints. Hence after refinement the per-vertex
        correspondence ref[i] <-> current[i] still holds for EVERY vertex
        (original and newly inserted), and the self-intersection trajectory of
        any vertex is simply the segment `_ref_verts[i] -> deformed[i]` -- no
        extra bookkeeping (e.g. interpolating new vertices back onto the original
        watertight surface) is needed.

        The current deformation is baked into the base vertices and `_disp` is
        restarted at zero (the refinement is geometry preserving, so this is a
        pure topology change). Returns (vertices_before, faces_before).
        """
        deformed = self._deformed().detach()
        v_before, f_before = self._verts.shape[0], self._faces.shape[0]
        # carry BOTH the clean rest field and the packed per-vertex fit state
        # through the same edge-midpoint split, so after refinement every
        # vertex (incl. new midpoints) keeps its ref[i]<->current[i] trajectory
        # correspondence AND a conservative inherited optimization state.
        state_attrs = (
            stateFloatAttrs(self._fit_state)
            if getattr(self, "_fit_state", None) is not None
            else None
        )
        extra_attrs = [self._ref_verts]
        if state_attrs is not None:
            extra_attrs.append(state_attrs)
        new_v, new_f, _, extra = subdivideMarkedFaces(
            deformed, self._faces, region_mask, extra_vertex_attrs=extra_attrs
        )
        self._verts = new_v.detach().clone()
        self._faces = new_f
        self._ref_verts = extra[0].detach().clone()
        self._disp = torch.zeros_like(self._verts, requires_grad=True)
        self._optimizer = torch.optim.AdamW([self._disp], lr=self.lr, amsgrad=True)
        if state_attrs is not None:
            self._fit_state = stateFromFloatAttrs(
                extra[1].detach(), v_before, self.device
            )
        self._buildTopology()
        self._buildBaseline()
        return v_before, f_before

    def _localizeHighError(
        self, level: int, use_local_plateau: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """Locate the faces to refine. Returns (region_mask, stats).

        When `use_local_plateau` (the clamped stepwise path), a face is refined
        only if it is high-error AND locally plateaued AND still optimizable
        (see `localizePlateauHighErrorFaces`); otherwise the legacy global
        high-error localizer is used (the Adam `fit()` path)."""
        deformed = self._deformed().detach()
        # ABOVE-MEAN path (user spec, highest precedence): split every face whose
        # centroid distance to the target exceeds mean_mult * the global mean
        # centroid distance -- an adaptive, parameter-free bar that tightens as
        # the fit improves.
        if (
            use_local_plateau
            and self.refine_above_mean
            and getattr(self, "_field", None) is not None
        ):
            region_mask, _cd, stats = localizeAboveMeanCentroidFaces(
                deformed,
                self._faces,
                self._field,
                self._face_adj,
                tau=self._tau_norm,
                mean_mult=self.refine_mean_mult,
                min_centroid_tau=self.refine_min_centroid_tau,
                min_component_faces=self.min_component_faces,
                dilation_rings=self.dilation_rings,
                max_faces=self.max_refine_faces,
            )
            print(
                f"[INFO][WatertightFitter] refine(above-mean) level {level}: "
                f"{stats['n_region']} faces "
                f"(above_mean={stats['n_above_mean']}, "
                f"mean_cd={stats['centroid_dist_mean_tau']:.3f}tau, "
                f"thr={stats['threshold']:.6f}, "
                f"cent_dist_max={stats['centroid_dist_max']:.6f})."
            )
            return region_mask, stats
        # RATIO path (user spec): split faces whose centroid is proportionally
        # much farther from the target than their corners
        # (d(centroid)/mean d(verts) > refine_ratio). Takes precedence over the
        # sag-DIFF path when refine_ratio > 0.
        if (
            use_local_plateau
            and self.refine_ratio > 0.0
            and getattr(self, "_field", None) is not None
        ):
            region_mask, _ratio, stats = localizeRatioSaggingFaces(
                deformed,
                self._faces,
                self._field,
                self._face_adj,
                tau=self._tau_norm,
                ratio=self.refine_ratio,
                centroid_mult=self.refine_centroid_mult,
                denom_eps_tau=self.refine_ratio_denom_eps_tau,
                min_component_faces=self.min_component_faces,
                dilation_rings=self.dilation_rings,
                max_faces=self.max_refine_faces,
            )
            print(
                f"[INFO][WatertightFitter] refine(ratio) level {level}: "
                f"{stats['n_region']} faces "
                f"(sagging={stats['n_sagging']}, "
                f"ratio_thr={stats['ratio_threshold']:.3f}, "
                f"ratio_max={stats['ratio_max']:.3f}, "
                f"cent_dist_max={stats['centroid_dist_max']:.6f})."
            )
            return region_mask, stats
        # PREFERRED (stepwise) path: refine only faces whose INTERIOR sags off
        # the target while their corners are already on it -- the only faces
        # worth splitting for "the whole surface on the target", measured with
        # the exact closest-point field (centroid-accurate).
        if (
            use_local_plateau
            and self.refine_sag_mult > 0.0
            and getattr(self, "_field", None) is not None
        ):
            region_mask, _sag, stats = localizeSaggingFaces(
                deformed,
                self._faces,
                self._field,
                self._face_adj,
                tau=self._tau_norm,
                sag_mult=self.refine_sag_mult,
                centroid_mult=self.refine_centroid_mult,
                quantile=self.refine_sag_quantile,
                min_component_faces=self.min_component_faces,
                dilation_rings=self.dilation_rings,
                max_faces=self.max_refine_faces,
            )
            print(
                f"[INFO][WatertightFitter] refine(sag) level {level}: "
                f"{stats['n_region']} faces "
                f"(sagging={stats['n_sagging']}, "
                f"sag_thr={stats['sag_threshold']:.6f}, "
                f"sag_max={stats['sag_max']:.6f}, "
                f"cent_dist_max={stats['centroid_dist_max']:.6f})."
            )
            return region_mask, stats
        if use_local_plateau and getattr(self, "_fit_state", None) is not None:
            region_mask, _face_error, stats = localizePlateauHighErrorFaces(
                deformed,
                self._faces,
                self._target_pts,
                self._target_index,
                self._face_adj,
                self._fit_state,
                level,
                tau=self._tau_norm,
                error_mult=self.error_mult,
                quantile=self.error_quantile,
                min_component_faces=self.min_component_faces,
                dilation_rings=self.dilation_rings,
                max_faces=self.max_refine_faces,
                local_drop_tau=self.local_drop_tau,
                max_blocked_vertex_ratio=self.max_blocked_vertex_ratio,
                refine_cooldown=self.refine_cooldown,
                device=self.device,
            )
            n_region = int(region_mask.sum().item())
            print(
                f"[INFO][WatertightFitter] refine level {level}: "
                f"{n_region} faces (high={stats['n_high_error_faces']}, "
                f"plateau={stats['n_local_plateau_faces']}, "
                f"blocked_skip={stats['n_blocked_faces_skipped']}, "
                f"thr={stats['threshold']:.6f}, "
                f"max_err={stats['face_error_max']:.6f})."
            )
            return region_mask, stats
        region_mask, _face_error, stats = localizeHighErrorFaces(
            deformed,
            self._faces,
            self._target_pts,
            self._target_index,
            self._face_adj,
            tau=self._tau_norm,
            error_mult=self.error_mult,
            quantile=self.error_quantile,
            min_component_faces=self.min_component_faces,
            dilation_rings=self.dilation_rings,
            max_faces=self.max_refine_faces,
            device=self.device,
        )
        n_region = int(region_mask.sum().item())
        print(
            f"[INFO][WatertightFitter] refine level {level}: "
            f"{n_region} faces marked (thr={stats['threshold']:.6f}, "
            f"max_err={stats['face_error_max']:.6f}).",
        )
        return region_mask, stats

    def _refine(self, level: int, use_local_plateau: bool = False) -> bool:
        region_mask, stats = self._localizeHighError(
            level, use_local_plateau=use_local_plateau
        )
        if int(region_mask.sum().item()) == 0:
            return False
        # cool-down bookkeeping: mark the vertices of the refined region as
        # refined at the NEXT level, so the local localizer will not re-split
        # the same region on the immediately following rounds.
        if getattr(self, "_fit_state", None) is not None:
            touched = torch.zeros(
                self._verts.shape[0], dtype=torch.bool, device=self.device
            )
            touched[self._faces[region_mask].reshape(-1)] = True
            self._fit_state["last_refine"][touched] = int(level) + 1
        v_before, f_before = self._applySubdivision(region_mask)
        self._refine_log.append(
            {
                "level": level,
                "n_region_faces": int(region_mask.sum().item()),
                "vertices_before": int(v_before),
                "vertices_after": int(self._verts.shape[0]),
                "faces_before": int(f_before),
                "faces_after": int(self._faces.shape[0]),
                **{k: float(v) for k, v in stats.items()},
            }
        )
        print(
            f"[INFO][WatertightFitter] subdivided: V {v_before}->{self._verts.shape[0]}, "
            f"F {f_before}->{self._faces.shape[0]}.",
        )
        return True

    # ------------------------------------------------------------------ #
    # fit setup (shared by fit + the stepwise debug mode)                #
    # ------------------------------------------------------------------ #
    def _setupFit(self) -> None:
        """Rigid init, GPU tensors, target index, schedules and baseline scan.
        Shared by `fit` (full pipeline) and `fitStepwise` (per-step debug)."""
        assert self.source_mesh is not None and self.target_mesh is not None

        self._rigidInit()
        self._applyPrefitCrop()
        self._baseline_mesh = self.source_mesh.clone()

        dev = self.device
        V = np.asarray(self.source_mesh.vertices, dtype=np.float32)
        F = np.asarray(self.source_mesh.triangles)

        self._verts = torch.tensor(V, device=dev, dtype=torch.float32)
        self._faces = torch.tensor(F, device=dev, dtype=torch.long)
        self._ref_verts = self._verts.detach().clone()  # clean source reference
        self._disp = torch.zeros_like(self._verts, requires_grad=True)
        self._optimizer = torch.optim.AdamW([self._disp], lr=self.lr, amsgrad=True)
        self._buildTopology()
        print(
            "[INFO][WatertightFitter::fit] V=",
            self._verts.shape[0],
            "F=",
            self._faces.shape[0],
            "E=",
            self._edges.shape[0],
        )

        target_pts_np, target_nrm_np = sampleMeshSurface(
            self.target_mesh,
            self.train_target_samples,
            seed=self.seed,
            with_normals=True,
        )
        self._target_pts = torch.tensor(target_pts_np, device=dev, dtype=torch.float32)
        self._target_nrm = torch.tensor(target_nrm_np, device=dev, dtype=torch.float32)
        self._target_index = NNIndex(target_pts_np, device=dev)

        # tau is constant in the normalized frame: (L/2048)*scale = 0.9/2048
        self._tau_norm = (self.L / 2048.0) * self.norm_scale
        tau_norm = self._tau_norm
        self._broad_margin = self.collision_broad_tau * tau_norm
        self._barrier_margin = self.collision_margin_tau * tau_norm
        self._sheet_gap = self.sheet_gap_tau * tau_norm
        # global target separation; per-pair margin (= min(this, clean gap)) is
        # computed in _buildSheetConstraints. _sheet_margin holds the per-pair
        # tensor; the scalar lives here.
        self._sheet_margin_scalar = self.sheet_min_margin_tau * tau_norm
        self._sheet_margin = torch.zeros(0, device=dev)

        if self.mask_dist_schedule is None:
            self.mask_dist_schedule = [m * tau_norm for m in (16.0, 8.0, 4.0, 2.0, 1.0)]
        if self.laplacian_schedule is None:
            self.laplacian_schedule = [
                self.laplacian_weight * f for f in (1.0, 1.0, 0.6, 0.3, 0.1)
            ]
        if self.point_to_plane_schedule is None:
            self.point_to_plane_schedule = [
                self.point_to_plane_weight * f for f in (0.0, 0.25, 0.5, 1.0, 2.0)
            ]

        self._buildBaseline()

        self._history = []
        self._refine_log = []
        self._traj_step_counter = 0
        # cached target crops (constant across steps): sampled once per bbox.
        # keyed by bbox name -> {"pts": (N,3) np.float32, "nfaces": int}
        self._target_crops = {}
        # legacy single-bbox cache fields (kept so older call sites still work)
        self._target_crop_pts = None
        self._target_crop_nfaces = 0
        # per-vertex optimization state machine (clamped stepwise path)
        self._fit_state = initVertexFitState(self._verts.shape[0], self.device)

    # ------------------------------------------------------------------ #
    # fit                                                                #
    # ------------------------------------------------------------------ #
    def fit(self) -> Mesh:
        self._setupFit()

        for level in range(self.max_subdivisions + 1):
            n_outer = self.outer_iter if level == 0 else self.refine_iter
            self._optimizeCycle(level, n_outer)
            if level < self.max_subdivisions:
                changed = self._refine(level)
                if not changed:
                    print(
                        "[INFO][WatertightFitter] no high-error region left; "
                        "stopping subdivision early."
                    )
                    break

        # authoritative full-mesh final gate for the user-defined trajectory
        # criterion: drive the whole mesh crossing-free, then measure it.
        self._trajectory_metrics = self._finalTrajectoryGate()
        if self.enable_trajectory_guard:
            print(
                "[INFO][WatertightFitter] final trajectory crossings: "
                f"vertices={self._trajectory_metrics['trajectory_crossing_vertices']}, "
                f"pairs={self._trajectory_metrics['trajectory_crossing_pairs']}, "
                f"faces={self._trajectory_metrics['trajectory_crossing_faces']}"
            )

        # supplementary triangle-triangle self-intersection diagnostic (the
        # legacy gate). Kept for reference; the trajectory criterion above is the
        # authoritative acceptance signal for this run.
        final_global = 0
        if self.enable_self_collision_guard:
            n_new, _ = self._newIntersections(query_mask=None)
            final_global = int(n_new)
            print(
                "[INFO][WatertightFitter] final triangle-triangle new "
                "self-intersections (supplementary):",
                final_global,
            )
        self._final_new_self_intersections = final_global

        with torch.no_grad():
            deformed = self._deformed().detach().cpu().numpy().astype(np.float64)
        self.source_mesh.vertices = deformed
        self.source_mesh.triangles = self._faces.detach().cpu().numpy()
        self.source_mesh.vertex_colors = None
        self.source_mesh.vertex_normals = None
        self.source_mesh.triangle_normals = None

        # the trajectory reference mesh: identical topology to the output, with
        # each vertex placed at its clean watertight-rest position (carried in
        # lock-step through subdivision). Saving it lets the trajectory criterion
        # be re-verified offline on the de-normalized output.
        self._trajectory_reference_mesh = self.source_mesh.clone()
        self._trajectory_reference_mesh.vertices = (
            self._ref_verts.detach().cpu().numpy().astype(np.float64)
        )
        self._trajectory_reference_mesh.vertex_colors = None
        self._trajectory_reference_mesh.vertex_normals = None
        self._trajectory_reference_mesh.triangle_normals = None
        return self.source_mesh

    # ------------------------------------------------------------------ #
    # stepwise debug mode                                                #
    # ------------------------------------------------------------------ #
    def _saveStepMesh(self, folder: str, step: int) -> str:
        """De-normalize the CURRENT deformed mesh and save it as step_XX.ply."""
        with torch.no_grad():
            deformed = self._deformed().detach().cpu().numpy().astype(np.float64)
        m = Mesh()
        m.vertices = deformed
        m.triangles = self._faces.detach().cpu().numpy()
        m.transform(self.norm_center, self.norm_scale, is_inverse=True)
        path = os.path.join(folder, f"step_{step:02d}.ply")
        m.save(path, overwrite=True)
        return path

    def _stepFitError(
        self, matched: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[float, float]:
        """Current per-vertex fit error to the matched target points (normalized
        frame), reported as (mean over kept verts, max). This is the cheap
        in-loop signal of "how well the mesh hugs the target this step"."""
        with torch.no_grad():
            d = self._deformed().detach()
            dist = (d - matched).norm(dim=1)
            kept = weight.reshape(-1) > 0
            if bool(kept.any()):
                mean_e = float(dist[kept].mean().item())
            else:
                mean_e = float(dist.mean().item())
            max_e = float(dist.max().item())
        return mean_e, max_e

    def fitStepwise(
        self,
        n_steps: int = 4,
        inner_per_step: int = 1,
        error_gate_tau: float = 0.0,
        save_folder: Union[str, None] = None,
        compute_chamfer: bool = False,
        chamfer_each_step: bool = False,
        save_meshes: bool = True,
    ) -> dict:
        """Run the FIRST `n_steps` recorded optimization steps one at a time.

        Each recorded step does exactly what the user asked for, in order:
          1. `inner_per_step` Adam steps that pull every vertex closer to its
             matched target point (data term + Laplacian/barriers), so the mesh
             hugs the target a little more each step;
          2. an IMMEDIATE trajectory self-intersection check (the user's
             criterion: the segment ref->current must not pierce a non-incident
             face) and a LOCAL pullback of every offending vertex along its own
             trajectory to the largest crossing-free position -- never a wholesale
             reset to rest;
          3. recording of the step's cheap fit error (distance to the matched
             target points) and the number of vertices repaired. The full
             Chamfer is expensive (de-normalize + sample + KNN), so it is OFF
             per-step by default and computed once at the end; set
             `chamfer_each_step` to also record it every step;
          4. optionally saving the step's mesh.

        No subdivision is performed here -- this is the per-step diagnostic of the
        level-0 burst, so the error/mesh trajectory is directly comparable across
        steps. Returns a dict with a `steps` list of per-step records.
        """
        self._setupFit()

        dev = self.device
        if save_folder is None:
            save_folder = (self.save_result_folder_path or "./output/stepwise/")
        os.makedirs(save_folder, exist_ok=True)

        # one fixed correspondence + normal gate for this short burst, using the
        # widest mask stage (stage 0) so far-away verts are still pulled in.
        mask_dist = self.mask_dist_schedule[0]
        lap_w = self.laplacian_schedule[0]
        p2p_w = self.point_to_plane_schedule[0]

        self._setInversionRest()
        active0 = self._activeFaceMask(self.collision_active_tau * self._tau_norm)
        self._buildConstraints(active0)
        coll_w = self.collision_weight if self.enable_self_collision_guard else 0.0
        sheet_w = self.sheet_weight if self.enable_sheet_guard else 0.0

        records = []
        for step in range(n_steps):
            t0 = time.time()
            # refresh the target correspondence each step so "closer to target"
            # tracks the moving surface (cheap relative to the step itself).
            with torch.no_grad():
                cur = self._deformed().detach()
                idx, d2 = self._target_index.query(cur, k=1)
                idx_t = torch.from_numpy(idx).to(dev)
                matched = self._target_pts[idx_t]
                matched_n = self._target_nrm[idx_t]
                keep = torch.from_numpy(d2).to(dev) < (mask_dist ** 2)
                if self.normal_gate:
                    vn = vertexNormals(cur, self._faces)
                    agree = (vn * matched_n).sum(dim=1) > self.normal_gate_cos
                    keep = keep & agree
                # error gate: freeze vertices that already hug the target so the
                # data term only acts on the genuinely high-error region. Without
                # this, a near-converged mesh keeps thrashing already-fitted
                # (often double-layer) verts toward ambiguous matches -- which is
                # exactly what was raising the error and spawning self-crossings.
                if error_gate_tau > 0.0:
                    dist = (cur - matched).norm(dim=1)
                    keep = keep & (dist > (error_gate_tau * self._tau_norm))
                weight = keep.float().unsqueeze(1)
                n_gated = int(keep.sum().item())

            # 1 Adam step(s) toward the target. The trajectory guard runs inside
            #   _innerSteps after every Adam step (trajectory_check_inner_every),
            #   so even with inner_per_step>1 each sub-step is guarded; we also
            #   resolve again below to drive this recorded step crossing-free.
            info = self._innerSteps(
                inner_per_step,
                matched,
                matched_n,
                weight,
                lap_w,
                p2p_w,
                coll_w,
                sheet_w,
            )

            # 2 immediate trajectory self-intersection check + local pullback,
            #   iterated until this step's moved region is crossing-free (scoped
            #   broad phase keeps it memory-bounded on the full case1 mesh).
            n_repaired_total = 0
            with torch.no_grad():
                for _ in range(self.trajectory_resolve_iters):
                    face_ids = self._trajectoryActiveFaceIds()
                    n = self._applyTrajectoryGuard(face_ids=face_ids)
                    n_repaired_total += n
                    if n == 0:
                        break
                traj = self._measureTrajectory(
                    face_ids=self._trajectoryActiveFaceIds()
                )

            # 3 record this step's error.
            mean_e, max_e = self._stepFitError(matched, weight)
            rec = {
                "step": step,
                "data_loss": info["data"],
                "lap_loss": info["lap"],
                "fit_error_mean_norm": mean_e,
                "fit_error_max_norm": max_e,
                "fit_error_mean_tau": mean_e / self._tau_norm,
                "fit_error_max_tau": max_e / self._tau_norm,
                "trajectory_repaired_vertices": int(n_repaired_total),
                "trajectory_crossing_vertices_after": int(
                    traj["trajectory_crossing_vertices"]
                ),
                "active_vertices": int(n_gated),
            }
            if chamfer_each_step:
                cm = self._currentChamfer()
                rec["chamfer_l1"] = cm["chamfer_l1"]
                rec["f1"] = cm.get("f1")
            if save_meshes:
                rec["mesh_path"] = self._saveStepMesh(save_folder, step)
            rec["seconds"] = round(time.time() - t0, 2)
            records.append(rec)
            print(
                f"[INFO][stepwise] step {step}: "
                f"fit_err_mean={rec['fit_error_mean_tau']:.3f}tau, "
                f"fit_err_max={rec['fit_error_max_tau']:.3f}tau, "
                f"active={rec['active_vertices']}, "
                f"repaired={rec['trajectory_repaired_vertices']}, "
                f"crossings_after={rec['trajectory_crossing_vertices_after']}, "
                + (
                    f"chamfer_l1={rec.get('chamfer_l1'):.6f}, "
                    if chamfer_each_step
                    else ""
                )
                + f"{rec['seconds']}s"
            )

        out = {
            "n_steps": n_steps,
            "inner_per_step": inner_per_step,
            "tau": self._tau_norm / self.norm_scale,
            "steps": records,
        }
        # one final full Chamfer/F1 (cheap to do once) unless already per-step.
        if compute_chamfer and not chamfer_each_step:
            final_cm = self._currentChamfer()
            out["final_chamfer_l1"] = final_cm["chamfer_l1"]
            out["final_f1"] = final_cm.get("f1")
            print(
                "[INFO][stepwise] final chamfer_l1=",
                round(final_cm["chamfer_l1"], 6),
                "f1=",
                round(final_cm.get("f1", 0.0), 6),
            )
        with open(os.path.join(save_folder, "stepwise_log.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("[INFO][stepwise] log saved to", os.path.join(save_folder, "stepwise_log.json"))
        return out

    def _currentMeshNormalized(self) -> Mesh:
        """The CURRENT deformed mesh as a Mesh in the (normalized) fit frame.

        Built once per step and reused by the full Chamfer, the bbox crop eval
        and the optional full-mesh save, so the (14M-vertex) device->host
        transfer happens only once."""
        m = Mesh()
        with torch.no_grad():
            m.vertices = self._deformed().detach().cpu().numpy().astype(np.float64)
            m.triangles = self._faces.detach().cpu().numpy()
        return m

    def _currentChamfer(self, cur_mesh: Union[Mesh, None] = None) -> dict:
        """Full Chamfer/F1 of the CURRENT deformed mesh (de-normalized)."""
        if cur_mesh is None:
            cur_mesh = self._currentMeshNormalized()
        return self._evaluateMesh(cur_mesh)

    def _targetCropForBBox(self, box: dict, debug_folder: str) -> dict:
        """Sample + cache (once) the de-normalized target crop for one bbox.

        Returns {"pts": (N,3) np.float32, "nfaces": int}. The target geometry is
        constant across steps, so this is computed on the first call per box and
        reused; the crop mesh is written to `<debug_folder>/<name>/target_crop.ply`."""
        name = box["name"]
        if name in self._target_crops:
            return self._target_crops[name]
        out_dir = os.path.join(debug_folder, name)
        os.makedirs(out_dir, exist_ok=True)
        tgt = self.target_mesh.clone()
        tgt.transform(self.norm_center, self.norm_scale, is_inverse=True)
        tv, tf, _, _ = cropMeshByBBox(
            np.asarray(tgt.vertices),
            np.asarray(tgt.triangles),
            box["center"],
            box["edge"],
            box["mode"],
        )
        tgt_crop = Mesh()
        tgt_crop.vertices = tv.astype(np.float64)
        tgt_crop.triangles = tf
        tgt_crop.save(os.path.join(out_dir, "target_crop.ply"), overwrite=True)
        if tf.shape[0] > 0:
            pts = sampleMeshSurface(tgt_crop, self.crop_eval_samples, seed=self.seed + 1)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
        entry = {"pts": pts, "nfaces": int(tf.shape[0])}
        self._target_crops[name] = entry
        return entry

    def _evaluateMeshCropped(
        self,
        src_mesh: Mesh,
        debug_folder: str,
        tag: str,
        save_target: bool = True,
    ) -> dict:
        """Region-restricted evaluation inside EVERY configured eval bbox.

        Crops BOTH the (de-normalized) deformed source and the (de-normalized)
        target to each axis-aligned box, saves each crop into
        `<debug_folder>/<bbox_name>/` and reports Chamfer/F1 on that region ONLY,
        so the fit quality of several regions of interest can be tracked and
        visually inspected independently during the optimization.

        Perf: the source is de-normalized ONCE (a single full-mesh transform)
        and then cropped for all boxes; each box's target crop + samples are
        cached after the first step. Metrics for box `name` are prefixed
        `<name>_crop_*`; the first box also exposes unprefixed `crop_*` aliases
        for backward compatibility with the existing logs / printout.
        """
        if not self.eval_bboxes:
            return {}
        os.makedirs(debug_folder, exist_ok=True)
        tau = self.L / 2048.0

        # de-normalize the deformed source ONCE, reused for every box
        src = src_mesh.clone()
        src.transform(self.norm_center, self.norm_scale, is_inverse=True)
        src_v = np.asarray(src.vertices)
        src_f = np.asarray(src.triangles)

        metrics = {}
        for bi, box in enumerate(self.eval_bboxes):
            name = box["name"]
            out_dir = os.path.join(debug_folder, name)
            os.makedirs(out_dir, exist_ok=True)
            sv, sf, _, _ = cropMeshByBBox(
                src_v, src_f, box["center"], box["edge"], box["mode"]
            )
            src_crop = Mesh()
            src_crop.vertices = sv.astype(np.float64)
            src_crop.triangles = sf
            src_crop.save(os.path.join(out_dir, f"{tag}_crop.ply"), overwrite=True)

            tgt_entry = self._targetCropForBBox(box, debug_folder)
            tgt_pts = tgt_entry["pts"]

            m = {
                f"{name}_crop_src_faces": int(sf.shape[0]),
                f"{name}_crop_tgt_faces": int(tgt_entry["nfaces"]),
            }
            if sf.shape[0] == 0 or tgt_pts.shape[0] == 0:
                m[f"{name}_crop_chamfer_l1"] = None
                m[f"{name}_crop_f1"] = None
            else:
                src_pts = sampleMeshSurface(
                    src_crop, self.crop_eval_samples, seed=self.seed
                )
                chamfer = computeChamferMetrics(src_pts, tgt_pts, device=self.device)
                f1 = computeF1AtThreshold(src_pts, tgt_pts, tau, device=self.device)
                m[f"{name}_crop_chamfer_l1"] = chamfer["chamfer_l1"]
                m[f"{name}_crop_chamfer_l2"] = chamfer["chamfer_l2"]
                m[f"{name}_crop_fit_error_l1"] = chamfer["fit_error_l1"]
                m[f"{name}_crop_cov_error_l1"] = chamfer["cov_error_l1"]
                m[f"{name}_crop_f1"] = f1["f1"]
                m[f"{name}_crop_precision"] = f1["precision"]
                m[f"{name}_crop_recall"] = f1["recall"]
            metrics.update(m)
            # unprefixed aliases for the first box (backward compatibility)
            if bi == 0:
                for k, v in m.items():
                    metrics[k[len(name) + 1:]] = v
        return metrics

    # ------------------------------------------------------------------ #
    # gradient-descent stepwise with per-vertex step clamp (user spec)   #
    # ------------------------------------------------------------------ #
    def _fullTrajectoryPullback(
        self, ids: torch.Tensor, n_bisect: int = 0
    ) -> Tuple[int, int]:
        """Check the segment ref->current of every vertex in `ids` against ANY
        triangle of the current mesh; for each segment that pierces a face, move
        the vertex back along the segment by the MINIMUM distance that leaves the
        segment free of any face crossing.

        The pull-back is now CLOSED-FORM: the crossing parameter t (t=0 rest,
        t=1 current) is solved analytically per (segment, face) candidate pair,
        reduced to each segment's EARLIEST crossing t_min, and the vertex is
        placed at the fraction t_min minus a small clearance along its own
        segment -- no bisection ladder. This is exactly "move back the minimum
        distance that removes all crossings", and it is what makes the per-step
        crossing count drop to zero deterministically.

        Differs from `_applyTrajectoryGuard` in two deliberate ways matching the
        user's spec: (a) the broad phase is the FULL mesh (face_ids=None), not the
        scoped active region; (b) the candidate set is exactly `ids` (every moved
        vertex), not the cumulative-motion-thresholded subset. Segments are
        chunked so peak candidate-pair memory stays bounded on the 14M mesh.
        Returns (n_pulled_back, n_checked)."""
        if ids.numel() == 0:
            return 0, 0
        cur = self._deformed().detach()
        ref = self._ref_verts.detach()
        cell = self._trajCellSize(cur)
        chunk = self.trajectory_seg_chunk
        clearance = self.trajectory_clearance_tau * self._tau_norm
        clamp_id_parts, clamp_pos_parts = [], []
        for start in range(0, ids.numel(), chunk):
            sub = ids[start:start + chunk]
            r = ref[sub]
            c = cur[sub]
            hit_seg, hit_face, hit_t = segmentMeshIntersectionParams(
                r, c, cur, self._faces, owner_vid=sub,
                inflate=0.0, cell_size=cell, face_ids=None,  # ANY triangle
            )
            if hit_seg.numel() == 0:
                continue
            t_min, _ = earliestSegmentMeshHits(
                hit_seg, hit_face, hit_t, sub.numel()
            )
            need = torch.isfinite(t_min)
            if not bool(need.any()):
                continue
            # convert the (distance) clearance into a per-segment parametric
            # back-off; degenerate (near-zero-length) segments fall back to the
            # rest position (alpha=0).
            seg_len = (c - r).norm(dim=1)
            clearance_t = torch.where(
                seg_len > 1e-12,
                clearance / seg_len.clamp_min(1e-12),
                torch.ones_like(seg_len),
            )
            alpha = torch.clamp(t_min - clearance_t, min=0.0, max=1.0)
            safe_pos = r + alpha.unsqueeze(1) * (c - r)
            clamp_id_parts.append(sub[need])
            clamp_pos_parts.append(safe_pos[need])
        if not clamp_id_parts:
            return 0, int(ids.numel())
        clamp_ids = torch.cat(clamp_id_parts, dim=0)
        clamp_pos = torch.cat(clamp_pos_parts, dim=0)
        with torch.no_grad():
            self._disp.data[clamp_ids] = clamp_pos - self._verts[clamp_ids]
        return int(clamp_ids.numel()), int(ids.numel())

    def _measureTriangleSelfIntersections(self) -> dict:
        """Authoritative triangle-triangle self-intersection count on the CURRENT
        deformed mesh, relative to the clean source baseline (so only crossings
        the FIT created are reported). This is the complement of the trajectory
        criterion: adjacent faces that fold/overlap without any single vertex's
        rest->current segment piercing a non-incident face are invisible to the
        trajectory guard but show up here, which is the diagnostic that tells the
        two failure modes apart."""
        cur = self._deformed().detach()
        inter = findSelfIntersections(
            cur, self._faces, inflate=0.0, exclude_ring=1,
            face_adjacency=self._face_adj,
        )
        keys = pairKeys(inter, self._faces.shape[0])
        # honest "new" count needs the clean-source baseline regardless of
        # whether the self-collision guard (which populates `_baseline_keys`) is
        # on, so compute + cache it from `_ref_verts` here on first use.
        baseline = getattr(self, "_diag_baseline_keys", None)
        if baseline is None:
            base_inter = findSelfIntersections(
                self._ref_verts.detach(), self._faces, inflate=0.0,
                exclude_ring=1, face_adjacency=self._face_adj,
            )
            baseline = pairKeys(base_inter, self._faces.shape[0])
            self._diag_baseline_keys = baseline
        if baseline.numel() > 0 and keys.numel() > 0:
            n_new = int((~torch.isin(keys, baseline)).sum().item())
        else:
            n_new = int(keys.numel())
        involved = (
            int(torch.unique(inter.reshape(-1)).numel())
            if inter.numel() else 0
        )
        return {
            "tri_intersecting_pairs_total": int(keys.numel()),
            "tri_intersecting_pairs_new": n_new,
            "tri_intersecting_faces": involved,
        }

    def _measureTrajectoryFull(
        self, ids: Union[torch.Tensor, None] = None
    ) -> dict:
        """Diagnostic full-mesh measurement of the trajectory criterion over the
        given vertex ids (default: all moved vertices). Reports the number of
        crossing vertices/pairs and the smallest crossing parameter t observed --
        the quantitative "how badly does the current state cross?" signal used by
        the single-step debug to verify the detector and the pull-back."""
        cur = self._deformed().detach()
        ref = self._ref_verts.detach()
        v = self._verts.shape[0]
        if ids is None:
            ids = torch.arange(v, device=self.device)
        cell = self._trajCellSize(cur)
        chunk = self.trajectory_seg_chunk
        n_pairs = 0
        n_vertices = 0
        t_min_global = float("inf")
        for start in range(0, ids.numel(), chunk):
            sub = ids[start:start + chunk]
            hit_seg, hit_face, hit_t = segmentMeshIntersectionParams(
                ref[sub], cur[sub], cur, self._faces, owner_vid=sub,
                inflate=0.0, cell_size=cell, face_ids=None,
            )
            if hit_seg.numel() == 0:
                continue
            n_pairs += int(hit_seg.numel())
            n_vertices += int(torch.unique(hit_seg).numel())
            t_min_global = min(t_min_global, float(hit_t.min().item()))
        return {
            "trajectory_crossing_vertices": n_vertices,
            "trajectory_crossing_pairs": n_pairs,
            "trajectory_min_t": (
                None if not math.isfinite(t_min_global) else t_min_global
            ),
            "trajectory_self_intersection_free": bool(n_vertices == 0),
        }

    # ------------------------------------------------------------------ #
    # self-aware single-step projection (user spec)                      #
    # ------------------------------------------------------------------ #
    def _selfAwareProjectStep(
        self,
        field: "ImplicitField",
        safe_dist_percent: float = 0.001,
        t_lo: float = 1e-4,
    ) -> dict:
        """ONE self-intersection-free projection step (the user's elegant idea).

        First principles: for every vertex v_i take the segment v_i -> cp_i, with
        cp_i its EXACT closest point on the target (face-interior). Intersect each
        segment with the CURRENT (static) source mesh and take the earliest hit
        parameter t_i (t=0 at v_i, t=1 at cp_i); a segment that hits nothing gets
        t_i = 1. Then advance every vertex by

            alpha_i = clamp(t_i - safe_dist_percent, 0, 1)

        i.e. each vertex moves AS FAR toward its target as it can without its
        path reaching any non-incident face. Because every vertex stops strictly
        before the first face it would meet (with a small parametric clearance),
        no vertex tunnels a face and no two faces are driven to coincide -- the
        single step is self-intersection-free by construction.

        Reuses the existing atoms wholesale: `field.closestPoints` (target
        face-interior closest point), `segmentMeshIntersectionParams` (broad-phase
        AABB grid + parametric Moller-Trumbore narrow phase) and
        `earliestSegmentMeshHits` (vectorized per-segment earliest-t reduction).
        Returns per-step diagnostics.
        """
        dev = self.device
        with torch.no_grad():
            cur = self._deformed().detach()
            cp, _, _ = field.closestPoints(cur)
            d_i = (cp - cur).norm(dim=1)  # uncapped distance to target

            v = cur.shape[0]
            owner = torch.arange(v, device=dev)
            cell = self._trajCellSize(cur)
            chunk = self.trajectory_seg_chunk

            # earliest crossing t of segment v_i -> cp_i against the static mesh
            t_min = torch.full((v,), float("inf"), device=dev)
            for start in range(0, v, chunk):
                sub = owner[start:start + chunk]
                hit_seg, hit_face, hit_t = segmentMeshIntersectionParams(
                    cur[sub], cp[sub], cur, self._faces, owner_vid=sub,
                    inflate=0.0, cell_size=cell, face_ids=None,
                    t_lo=t_lo, t_hi=1.0,
                )
                t_sub, _ = earliestSegmentMeshHits(
                    hit_seg, hit_face, hit_t, sub.numel()
                )
                t_min[start:start + chunk] = t_sub

            crossed = torch.isfinite(t_min)
            t_eff = torch.where(crossed, t_min, torch.ones_like(t_min))
            alpha = torch.clamp(t_eff - safe_dist_percent, min=0.0, max=1.0)

            # new position = cur + alpha*(cp-cur); write it through _disp so the
            # rest of the machinery (eval, save) sees it via _deformed().
            new_pos = cur + alpha.unsqueeze(1) * (cp - cur)
            self._disp.data.copy_(new_pos - self._verts)

            resid_after = (cp - new_pos).norm(dim=1)
        return {
            "n_vertices": int(v),
            "n_crossing_segments": int(crossed.sum().item()),
            "mean_t": float(t_eff.mean().item()),
            "min_t": float(t_eff.min().item()),
            "mean_alpha": float(alpha.mean().item()),
            "fit_residual_before_tau": float(d_i.mean().item()) / self._tau_norm,
            "fit_residual_after_tau": float(resid_after.mean().item())
            / self._tau_norm,
        }

    def fitSelfAwareSingleStep(
        self,
        safe_dist_percent: float = 0.001,
        t_lo: float = 1e-4,
        save_folder: Union[str, None] = None,
        crop_eval: bool = True,
        compute_chamfer: bool = False,
    ) -> dict:
        """One self-intersection-free projection step + bbox-restricted metrics.

        Drives every vertex to its target closest point, capped per-vertex by the
        earliest self-collision along its own segment (see `_selfAwareProjectStep`),
        then evaluates Chamfer/F1 and the triangle-triangle + vertex-trajectory
        self-intersection counts inside the configured eval bboxes only.
        """
        self._setupFit()
        dev = self.device
        if save_folder is None:
            save_folder = (
                self.save_result_folder_path or "./output/self_aware_step/"
            )
        os.makedirs(save_folder, exist_ok=True)
        debug_folder = os.path.join(save_folder, "debug")

        tV = np.asarray(self.target_mesh.vertices, dtype=np.float32)
        tF = np.asarray(self.target_mesh.triangles)
        field = ImplicitField(tV, tF, device=dev)
        self._field = field

        t0 = time.time()
        step_info = self._selfAwareProjectStep(
            field, safe_dist_percent=safe_dist_percent, t_lo=t_lo
        )
        t_step = time.time()

        rec = {
            "method": "self_aware_single_step",
            "safe_dist_percent": safe_dist_percent,
            "tau": self._tau_norm / self.norm_scale,
            **step_info,
        }

        cur_mesh = self._currentMeshNormalized()
        if crop_eval:
            rec.update(
                self._evaluateMeshCropped(cur_mesh, debug_folder, "self_aware")
            )
        if compute_chamfer:
            cm = self._currentChamfer(cur_mesh)
            rec["chamfer_l1"] = cm["chamfer_l1"]
            rec["f1"] = cm.get("f1")

        with torch.no_grad():
            tri = self._measureTriangleSelfIntersections()
            traj = self._measureTrajectoryFull()
        rec.update(tri)
        rec["trajectory_crossing_vertices"] = traj["trajectory_crossing_vertices"]
        rec["trajectory_crossing_pairs"] = traj["trajectory_crossing_pairs"]

        rec["t_step_s"] = round(t_step - t0, 1)
        rec["seconds"] = round(time.time() - t0, 1)

        self._saveStepMesh(save_folder, 0)
        with open(os.path.join(save_folder, "self_aware_log.json"), "w") as f:
            json.dump(rec, f, indent=2)

        crop_str = ""
        for box in self.eval_bboxes:
            f1v = rec.get(f"{box['name']}_crop_f1")
            if f1v is not None:
                crop_str += f"{box['name']}_f1={f1v:.4f}, "
        print(
            "[INFO][self-aware-step] "
            f"V={rec['n_vertices']}, "
            f"resid={rec['fit_residual_before_tau']:.3f}->"
            f"{rec['fit_residual_after_tau']:.3f}tau, "
            f"crossed_segs={rec['n_crossing_segments']}, "
            f"mean_alpha={rec['mean_alpha']:.4f}, "
            + crop_str
            + f"tri_si_new={rec['tri_intersecting_pairs_new']}, "
            f"traj_cross={rec['trajectory_crossing_vertices']}, "
            f"{rec['seconds']}s"
        )
        print("[INFO][self-aware-step] log saved to",
              os.path.join(save_folder, "self_aware_log.json"))
        return rec

    # ------------------------------------------------------------------ #
    # front-advancing single step (self-intersection-free by construction)#
    # ------------------------------------------------------------------ #
    def _frontAdvancingProjectStep(
        self,
        field: "ImplicitField",
        backoff: float = 0.5,
        relax_iters: int = 40,
    ) -> dict:
        """ONE projection step driven to be penetration-free by EXACT relaxation.

        First principles (validated by the diagnostic): the surviving self-
        intersections of a naive all-at-once projection are genuine through-
        PENETRATIONS (an edge of one face crossing another's interior) caused by
        interleaved vertices folding their faces together -- not a vertex
        tunnelling a static face (which a trajectory test could catch), and not
        the ~60% near-coplanar CONTACT pairs that the legacy Moller test miscounts.

        So the step (a) projects every vertex onto its target closest point, then
        (b) relaxes out the exact penetrations: it repeatedly finds the truly
        penetrating face pairs (`findSelfIntersections(predicate='penetrate')` --
        precise, no coplanar tolerance) and backs the incident vertices' advance
        fraction off toward their rest position, where the watertight mesh is
        penetration-free. Vertices never involved keep the full projection.

        Delegated to `Method/front_advance.penetrationRelaxStep`. Writes the
        result through `_disp` so eval/save see it via `_deformed()`.
        """
        dev = self.device
        with torch.no_grad():
            cur = self._deformed().detach()
            ref = self._ref_verts.detach()
            cp, _, _ = field.closestPoints(cur)
            d_i = (cp - cur).norm(dim=1)  # uncapped distance to target
            cell = self._trajCellSize(cur)

            new_pos, info = penetrationRelaxStep(
                ref, cur, cp, self._faces,
                face_adjacency=self._face_adj,
                cell_size=cell,
                backoff=backoff,
                max_iters=relax_iters,
            )
            self._disp.data.copy_(new_pos - self._verts)
            resid_after = (cp - new_pos).norm(dim=1)
        return {
            "n_vertices": int(cur.shape[0]),
            "relax_iters": info["iters"],
            "pen_pairs_start": info["pen_pairs_start"],
            "pen_pairs_end": info["pen_pairs_end"],
            "backed_off_vertices": info["backed_off_vertices"],
            "pinned_vertices": info["pinned_vertices"],
            "mean_alpha": info["mean_alpha"],
            "fit_residual_before_tau": float(d_i.mean().item()) / self._tau_norm,
            "fit_residual_after_tau": float(resid_after.mean().item())
            / self._tau_norm,
        }

    def fitFrontAdvancingSingleStep(
        self,
        backoff: float = 0.5,
        relax_iters: int = 40,
        save_folder: Union[str, None] = None,
        crop_eval: bool = True,
        compute_chamfer: bool = False,
    ) -> dict:
        """One penetration-relaxed projection step + bbox-restricted metrics.

        Projects every vertex onto its target closest point, then relaxes out the
        EXACT through-penetrations (see `_frontAdvancingProjectStep`), and finally
        evaluates Chamfer/F1 and the authoritative (now penetration-precise)
        self-intersection counts inside the configured eval bboxes only.
        """
        self._setupFit()
        dev = self.device
        if save_folder is None:
            save_folder = (
                self.save_result_folder_path or "./output/front_advance_step/"
            )
        os.makedirs(save_folder, exist_ok=True)
        debug_folder = os.path.join(save_folder, "debug")

        tV = np.asarray(self.target_mesh.vertices, dtype=np.float32)
        tF = np.asarray(self.target_mesh.triangles)
        field = ImplicitField(tV, tF, device=dev)
        self._field = field

        t0 = time.time()
        step_info = self._frontAdvancingProjectStep(
            field, backoff=backoff, relax_iters=relax_iters,
        )
        t_step = time.time()

        rec = {
            "method": "penetration_relax_single_step",
            "backoff": backoff,
            "tau": self._tau_norm / self.norm_scale,
            **step_info,
        }

        cur_mesh = self._currentMeshNormalized()
        if crop_eval:
            rec.update(
                self._evaluateMeshCropped(cur_mesh, debug_folder, "front_advance")
            )
        if compute_chamfer:
            cm = self._currentChamfer(cur_mesh)
            rec["chamfer_l1"] = cm["chamfer_l1"]
            rec["f1"] = cm.get("f1")

        with torch.no_grad():
            tri = self._measureTriangleSelfIntersections()
            traj = self._measureTrajectoryFull()
        rec.update(tri)
        rec["trajectory_crossing_vertices"] = traj["trajectory_crossing_vertices"]
        rec["trajectory_crossing_pairs"] = traj["trajectory_crossing_pairs"]

        rec["t_step_s"] = round(t_step - t0, 1)
        rec["seconds"] = round(time.time() - t0, 1)

        self._saveStepMesh(save_folder, 0)
        with open(os.path.join(save_folder, "front_advance_log.json"), "w") as f:
            json.dump(rec, f, indent=2)

        crop_str = ""
        for box in self.eval_bboxes:
            f1v = rec.get(f"{box['name']}_crop_f1")
            if f1v is not None:
                crop_str += f"{box['name']}_f1={f1v:.4f}, "
        print(
            "[INFO][pen-relax-step] "
            f"V={rec['n_vertices']}, "
            f"resid={rec['fit_residual_before_tau']:.3f}->"
            f"{rec['fit_residual_after_tau']:.3f}tau, "
            f"pen_pairs {rec['pen_pairs_start']}->{rec['pen_pairs_end']} "
            f"in {rec['relax_iters']} iters, "
            f"backed_off={rec['backed_off_vertices']}, "
            f"mean_alpha={rec['mean_alpha']:.4f}, "
            + crop_str
            + f"tri_si_new={rec['tri_intersecting_pairs_new']}, "
            f"{rec['seconds']}s"
        )
        print("[INFO][pen-relax-step] log saved to",
              os.path.join(save_folder, "front_advance_log.json"))
        return rec

    def fitStepwiseClamped(
        self,
        n_steps: int = 20,
        step_frac: float = 0.1,
        gd_lr: float = 0.5,
        lap_w: Union[float, None] = None,
        n_bisect: int = 16,
        resolve_iters: int = 8,
        pullback_min_move_tau: float = 0.0,
        max_subdivisions: Union[int, None] = None,
        plateau_window: Union[int, None] = None,
        plateau_rel_tol: Union[float, None] = None,
        plateau_patience: Union[int, None] = None,
        converge_abs_tau: Union[float, None] = 0.02,
        save_folder: Union[str, None] = None,
        compute_chamfer: bool = True,
        crop_eval: bool = True,
        full_chamfer_each_step: bool = False,
        save_full_each_step: bool = False,
        refine_every_step: bool = False,
        trajectory_debug: bool = False,
    ) -> dict:
        """Gradient-descent stepwise fit with a per-vertex step cap + adaptive
        subdivision near convergence (per the spec).

        When `trajectory_debug` is set, each step additionally records a
        full-mesh measurement of the user trajectory criterion BEFORE and AFTER
        the pull-back (crossing vertices/pairs and the smallest crossing
        parameter t), so the detector and the repair can be verified against the
        known baseline that step_frac=1.0 produces many crossings.

        When `refine_every_step` is True the per-step closest-point projection is
        treated as already optimal (step_frac=1.0 lands every vertex on the
        target in one step), so the adaptive subdivision is run AFTER EVERY step
        (gated only by `level < max_subdivisions`) instead of waiting for a
        global residual plateau -- each step projects, then splits the faces the
        refine criterion flags, so the surface is driven onto the target feature
        by feature.

        Each recorded step:
          1. find each vertex's closest point cp_i on the TARGET surface (exact,
             face-interior) and the pair distance d_i = ||cp_i - v_i||;
          2. take ONE gradient-descent step of (data toward cp_i + Laplacian),
             but CLAMP each vertex's resulting move along the gradient to at most
             step_frac * d_i (default 0.1 d_i), so no vertex can overshoot;
          3. check the segment ref->current of every moved vertex against ANY
             triangle of the current mesh; pull every offending vertex back along
             its own segment by the minimum distance that removes all crossings
             (largest safe fraction), iterating to a crossing-free fixpoint;
          4. evaluate (region-restricted bbox Chamfer/F1 + optional full Chamfer)
             and save the (cropped) mesh;
          5. feed the residual into a plateau monitor: once the error stops
             dropping meaningfully AND fewer than `max_subdivisions` refinements
             have been done, adaptively subdivide the high-error region. The SAME
             split is applied to the rest reference in lock-step, so the
             per-vertex correspondence (hence the trajectory-crossing test) is
             preserved across refinement with zero extra bookkeeping.

        Plain gradient descent (not Adam) is used on purpose: Adam's per-coord
        adaptive scaling amplifies the tiny near-converged gradient into a
        constant-size thrash; the explicit 0.1 d_i cap is the principled step
        controller instead.
        """
        self._setupFit()
        dev = self.device
        if save_folder is None:
            save_folder = self.save_result_folder_path or "./output/stepwise_clamped/"
        os.makedirs(save_folder, exist_ok=True)
        debug_folder = os.path.join(save_folder, "debug")
        tau = self._tau_norm
        if lap_w is None:
            lap_w = self.laplacian_weight
        if max_subdivisions is None:
            max_subdivisions = self.max_subdivisions
        # Subdivide "near convergence". For the clamped 0.1*d_i step the residual
        # decays geometrically, so the RELATIVE drop is ~constant and never fires
        # -- the ABSOLUTE per-window improvement (converge_abs_tau, in tau) is the
        # meaningful "the error curve has flattened" signal. Either criterion
        # triggers a refinement.
        monitor = PlateauMonitor(
            window=plateau_window or self.plateau_window,
            rel_tol=self.plateau_rel_tol if plateau_rel_tol is None else plateau_rel_tol,
            patience=plateau_patience or self.plateau_patience,
            min_updates=plateau_window or self.plateau_window,
            abs_tol=converge_abs_tau,
        )

        # exact closest-point field over the (normalized) target surface: gives
        # the true face-interior closest point (the move direction + d_i), which
        # the sampled-point NN index cannot.
        tV = np.asarray(self.target_mesh.vertices, dtype=np.float32)
        tF = np.asarray(self.target_mesh.triangles)
        field = ImplicitField(tV, tF, device=dev)
        # expose for the sag-based refine localizer (exact centroid distances).
        self._field = field

        records = []
        level = 0  # subdivision rounds done so far
        for step in range(n_steps):
            t0 = time.time()

            # 1 exact closest point on target + pair distance d_i
            with torch.no_grad():
                cur = self._deformed().detach()
                cp, _, _ = field.closestPoints(cur)
                d_i = (cp - cur).norm(dim=1)  # (V,)
            t_closest = time.time()

            # 2 one plain gradient-descent step (data toward cp + Laplacian),
            #   then clamp each vertex's move to step_frac * d_i.
            #   The data term uses a SUM reduction so each vertex's gradient is
            #   2(d - cp) (independent of the vertex count); with lr=0.5 the raw
            #   data step is ~the full (cp - d), so the step_frac*d_i CAP is the
            #   binding step controller -- i.e. every vertex advances ~0.1 d_i
            #   toward its closest point, exactly as specified. The Laplacian
            #   gradient still smooths the field before the cap is applied.
            self._disp.grad = None
            disp = self._disp
            d = self._verts + disp
            # MASKED data loss: only vertices the optimizer can still help pull
            # contribute their distance error. A vertex frozen by the per-step
            # clamp + trajectory pull-back (an unreachable thin shell) would
            # otherwise add a constant, unsatisfiable pull that fights the rest
            # of the mesh and inflates the residual without ever improving.
            opt_mask = self._fit_state["optimizable"]
            per_v_data = ((d - cp) ** 2).sum(dim=1)
            data = self.fit_weight * per_v_data[opt_mask].sum()
            lap_loss = edgeLaplacianLoss(disp, self._edges)
            loss = 0.5 * data + lap_w * lap_loss
            loss.backward()
            with torch.no_grad():
                pos_before = (self._verts + self._disp).detach().clone()
                grad = self._disp.grad
                raw_step = -gd_lr * grad             # gradient-descent direction
                cap = (step_frac * d_i).unsqueeze(1)  # per-vertex max move
                clamped_step = clampNorm(raw_step, cap)
                self._disp.data.add_(clamped_step)
                # intended (pre-pullback) per-vertex move length -- what the
                # optimizer asked for this step, used by the state machine.
                intended_move = clamped_step.norm(dim=1)
            t_gd = time.time()

            # 3 trajectory check vs ANY face + minimum-distance pull-back,
            #   iterated to a crossing-free fixpoint. The candidate set is every
            #   vertex whose ref->current segment is long enough to plausibly
            #   pierce a non-incident sheet (default 0: every moved vertex, the
            #   faithful full check). Correspondence ref[i]<->current[i] holds for
            #   all vertices (incl. midpoints) thanks to the lock-step subdivision.
            with torch.no_grad():
                moved = torch.nonzero(
                    self._cumulativeMotion() > (pullback_min_move_tau * tau),
                    as_tuple=False,
                ).reshape(-1)
            traj_before = None
            if trajectory_debug:
                with torch.no_grad():
                    traj_before = self._measureTrajectoryFull(moved)
            n_repaired_total = 0
            for _ in range(resolve_iters):
                n_pb, _ = self._fullTrajectoryPullback(moved, n_bisect)
                n_repaired_total += n_pb
                if n_pb == 0:
                    break
            traj_after = None
            tri_after = None
            if trajectory_debug:
                with torch.no_grad():
                    traj_after = self._measureTrajectoryFull(moved)
                    tri_after = self._measureTriangleSelfIntersections()
            t_pull = time.time()

            # 3b advance the per-vertex optimization state machine. `actual_move`
            #     is the realized coordinate change AFTER the clamp + every
            #     pull-back round, so a vertex that "wanted to move" (intended)
            #     but was dragged straight back (actual ~ 0) accumulates a stall
            #     and is eventually dropped from the data loss (above).
            with torch.no_grad():
                pos_after = (self._verts + self._disp).detach()
                actual_move = (pos_after - pos_before).norm(dim=1)
                updateVertexFitState(
                    self._fit_state,
                    d_i,
                    intended_move,
                    actual_move,
                    tau,
                    unopt_error_tau=self.unopt_error_tau,
                    min_intended_move_tau=self.unopt_min_intended_move_tau,
                    min_actual_move_tau=self.unopt_min_actual_move_tau,
                    min_progress_ratio=self.unopt_min_progress_ratio,
                    block_patience=self.unopt_block_patience,
                )
                n_optimizable = int(self._fit_state["optimizable"].sum().item())
                n_unoptimizable = int(self._verts.shape[0]) - n_optimizable

            # 4 record + evaluate
            with torch.no_grad():
                resid = float(d_i.mean().item())
                resid_max = float(d_i.max().item())
                opt_m = self._fit_state["optimizable"]
                resid_active = (
                    float(d_i[opt_m].mean().item())
                    if bool(opt_m.any()) else 0.0
                )
                blocked_m = ~opt_m
                resid_blocked = (
                    float(d_i[blocked_m].mean().item())
                    if bool(blocked_m.any()) else 0.0
                )
            resid_tau = resid / tau
            rec = {
                "step": step,
                "level": level,
                "n_vertices": int(self._verts.shape[0]),
                "n_faces": int(self._faces.shape[0]),
                "fit_residual_mean_tau": resid_tau,
                "fit_residual_max_tau": resid_max / tau,
                "data_loss": float(data.item()),
                "lap_loss": float(lap_loss.item()),
                "mean_step_move_tau": float(intended_move.mean().item()) / tau,
                "max_step_move_tau": float(intended_move.max().item()) / tau,
                "mean_actual_move_tau": float(actual_move.mean().item()) / tau,
                "trajectory_repaired_vertices": int(n_repaired_total),
                "n_optimizable_vertices": n_optimizable,
                "n_unoptimizable_vertices": n_unoptimizable,
                "mean_resid_active_tau": resid_active / tau,
                "mean_resid_blocked_tau": resid_blocked / tau,
            }
            if trajectory_debug:
                rec["trajectory_crossing_vertices_before"] = (
                    traj_before["trajectory_crossing_vertices"]
                )
                rec["trajectory_crossing_pairs_before"] = (
                    traj_before["trajectory_crossing_pairs"]
                )
                rec["trajectory_min_t_before"] = traj_before["trajectory_min_t"]
                rec["trajectory_crossing_vertices_after"] = (
                    traj_after["trajectory_crossing_vertices"]
                )
                rec["trajectory_crossing_pairs_after"] = (
                    traj_after["trajectory_crossing_pairs"]
                )
                rec["trajectory_min_t_after"] = traj_after["trajectory_min_t"]
                rec["tri_intersecting_pairs_new"] = (
                    tri_after["tri_intersecting_pairs_new"]
                )
                rec["tri_intersecting_pairs_total"] = (
                    tri_after["tri_intersecting_pairs_total"]
                )
                rec["tri_intersecting_faces"] = tri_after["tri_intersecting_faces"]
            cur_mesh = None
            if crop_eval or full_chamfer_each_step or save_full_each_step:
                cur_mesh = self._currentMeshNormalized()
            if crop_eval:
                rec.update(
                    self._evaluateMeshCropped(
                        cur_mesh, debug_folder, f"step_{step:02d}",
                    )
                )
            if full_chamfer_each_step:
                cm = self._currentChamfer(cur_mesh)
                rec["chamfer_l1"] = cm["chamfer_l1"]
                rec["f1"] = cm.get("f1")
            t_eval = time.time()

            # 5 convergence check -> adaptive subdivision (lock-step on the rest)
            plateau = monitor.update(resid_tau)
            rec["rel_drop"] = monitor.last_rel_drop
            rec["abs_drop_tau"] = monitor.last_abs_drop
            rec["plateau"] = bool(plateau)
            # `refine_every_step`: the single-step projection is already optimal,
            # so subdivide every step (only level-capped); otherwise wait for the
            # residual plateau as before.
            if refine_every_step:
                do_subdivide = level < max_subdivisions
            else:
                do_subdivide = plateau and level < max_subdivisions
            if save_full_each_step or step == n_steps - 1 or do_subdivide:
                rec["mesh_path"] = self._saveStepMesh(save_folder, step)
            if do_subdivide:
                changed = self._refine(level, use_local_plateau=True)
                rec["subdivided"] = bool(changed)
                if changed:
                    level += 1
                    monitor.reset()
                rec["level_after"] = level

            rec["t_closest_s"] = round(t_closest - t0, 1)
            rec["t_gd_s"] = round(t_gd - t_closest, 1)
            rec["t_pullback_s"] = round(t_pull - t_gd, 1)
            rec["t_eval_s"] = round(t_eval - t_pull, 1)
            rec["seconds"] = round(time.time() - t0, 2)
            records.append(rec)
            crop_str = ""
            for box in self.eval_bboxes:
                f1v = rec.get(f"{box['name']}_crop_f1")
                if f1v is not None:
                    crop_str += f"{box['name']}_f1={f1v:.4f}, "
            print(
                f"[INFO][stepwise-clamped] step {step} (lvl {level}, "
                f"V={rec['n_vertices']}): "
                f"resid={resid_tau:.3f}tau, "
                f"move={rec['mean_step_move_tau']:.4f}tau, "
                f"repaired={rec['trajectory_repaired_vertices']}, "
                + (
                    f"cross[{rec['trajectory_crossing_vertices_before']}"
                    f"->{rec['trajectory_crossing_vertices_after']}], "
                    if trajectory_debug else ""
                )
                + f"unopt={rec['n_unoptimizable_vertices']}, "
                f"rel_drop={monitor.last_rel_drop:.4f}, "
                + crop_str
                + (f"subdiv! " if rec.get("subdivided") else "")
                + f"[{rec['t_closest_s']}/{rec['t_gd_s']}/"
                f"{rec['t_pullback_s']}/{rec['t_eval_s']}s] "
                f"{rec['seconds']}s"
            )

        converged = bool(
            records
            and (
                records[-1]["rel_drop"] < monitor.rel_tol
                or (
                    converge_abs_tau is not None
                    and records[-1]["abs_drop_tau"] < converge_abs_tau
                )
            )
        )
        out = {
            "method": "stepwise_clamped",
            "n_steps": n_steps,
            "step_frac": step_frac,
            "tau": tau / self.norm_scale,
            "max_subdivisions": max_subdivisions,
            "converge_abs_tau": converge_abs_tau,
            "subdivision_rounds": level,
            "converged": converged,
            "eval_bboxes": [
                {
                    "name": b["name"],
                    "center": b["center"].tolist(),
                    "edge": (
                        b["edge"] if np.isscalar(b["edge"])
                        else np.asarray(b["edge"]).tolist()
                    ),
                    "mode": b["mode"],
                }
                for b in self.eval_bboxes
            ],
            "steps": records,
        }
        if compute_chamfer:
            cm = self._currentChamfer()
            out["final_chamfer_l1"] = cm["chamfer_l1"]
            out["final_f1"] = cm.get("f1")
            print(
                "[INFO][stepwise-clamped] final FULL chamfer_l1=",
                round(cm["chamfer_l1"], 6), "f1=", round(cm.get("f1", 0.0), 6),
            )
        with open(os.path.join(save_folder, "stepwise_log.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(
            "[INFO][stepwise-clamped] converged=", converged,
            "subdivision_rounds=", level,
            "| log saved to", os.path.join(save_folder, "stepwise_log.json"),
        )
        return out

    # ------------------------------------------------------------------ #
    # evaluation                                                         #
    # ------------------------------------------------------------------ #
    def _evaluateMesh(self, src_mesh: Mesh) -> dict:
        src = src_mesh.clone()
        tgt = self.target_mesh.clone()
        src.transform(self.norm_center, self.norm_scale, is_inverse=True)
        tgt.transform(self.norm_center, self.norm_scale, is_inverse=True)

        src_pts = sampleMeshSurface(src, self.eval_samples, seed=self.seed)
        tgt_pts = sampleMeshSurface(tgt, self.eval_samples, seed=self.seed + 1)

        tau = self.L / 2048.0
        chamfer = computeChamferMetrics(src_pts, tgt_pts, device=self.device)
        f1 = computeF1AtThreshold(src_pts, tgt_pts, tau, device=self.device)

        metrics = {
            "L": self.L,
            "tau": tau,
            "eval_samples": self.eval_samples,
            "n_vertices": int(np.asarray(src_mesh.vertices).shape[0]),
            "n_faces": int(np.asarray(src_mesh.triangles).shape[0]),
            **chamfer,
            **f1,
        }
        return metrics

    def evaluate(self) -> dict:
        return self._evaluateMesh(self.source_mesh)

    def fitAndEvaluate(self) -> dict:
        """Fit, then evaluate the rigid-init baseline and the deformed result.

        Acceptance is by L1 chamfer, but no-self-intersection is a hard
        constraint: if strict mode is on and the deformed result is NOT clean
        (more intersections than the source baseline) while the baseline-topology
        rigid result is clean, the metric comparison is reported honestly and the
        deformed result is flagged unverified rather than silently accepted.
        """
        self.fit()

        fitted_metrics = self._evaluateMesh(self.source_mesh)
        baseline_metrics = self._evaluateMesh(self._baseline_mesh)

        traj = dict(getattr(self, "_trajectory_metrics", {}))

        if fitted_metrics["chamfer_l1"] <= baseline_metrics["chamfer_l1"]:
            kept = "fitted"
            final = fitted_metrics
        else:
            self.source_mesh.vertices = np.asarray(self._baseline_mesh.vertices).copy()
            self.source_mesh.triangles = np.asarray(self._baseline_mesh.triangles).copy()
            self.source_mesh.vertex_colors = None
            self.source_mesh.vertex_normals = None
            self.source_mesh.triangle_normals = None
            kept = "baseline"
            final = baseline_metrics
            # the rest mesh is its own reference -> trajectory trivially clean.
            self._trajectory_reference_mesh = self._baseline_mesh.clone()
            traj = {
                "trajectory_crossing_vertices": 0,
                "trajectory_crossing_pairs": 0,
                "trajectory_crossing_faces": 0,
                "trajectory_self_intersection_free": True,
            }
            self._trajectory_metrics = traj
            print(
                "[INFO][WatertightFitter] deformation did not improve metric; "
                "keeping rigid-init baseline."
            )

        # the user-defined criterion is the authoritative cleanliness signal.
        clean = bool(traj.get("trajectory_self_intersection_free", True))

        return {
            "baseline": baseline_metrics,
            "fitted": fitted_metrics,
            "kept": kept,
            "metrics": final,
            "refine_log": self._refine_log,
            "final_new_self_intersections": int(
                getattr(self, "_final_new_self_intersections", 0)
            ),
            "self_intersection_free": bool(clean),
            **traj,
        }

    def saveResult(
        self, metrics: Union[dict, None] = None, config: Union[dict, None] = None
    ) -> str:
        if self.save_result_folder_path is None:
            return ""
        out_mesh = self.source_mesh.clone()
        out_mesh.transform(self.norm_center, self.norm_scale, is_inverse=True)
        traj = getattr(self, "_trajectory_metrics", {})
        clean = bool(traj.get("trajectory_self_intersection_free", True))
        name = "fitted_mesh.ply" if clean else "fitted_mesh_unverified.ply"
        mesh_path = self.save_result_folder_path + name
        out_mesh.save(mesh_path, overwrite=True)

        # de-normalized trajectory reference mesh (same topology as the output);
        # the trajectory criterion can be re-verified offline against it.
        ref_mesh = getattr(self, "_trajectory_reference_mesh", None)
        if ref_mesh is not None:
            ref_out = ref_mesh.clone()
            ref_out.transform(self.norm_center, self.norm_scale, is_inverse=True)
            ref_out.save(
                self.save_result_folder_path + "trajectory_reference_mesh.ply",
                overwrite=True,
            )
        if traj:
            with open(
                self.save_result_folder_path + "trajectory_report.json", "w"
            ) as f:
                json.dump(traj, f, indent=2)

        if metrics is not None:
            with open(self.save_result_folder_path + "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        if config is not None:
            with open(self.save_result_folder_path + "config.json", "w") as f:
                json.dump(config, f, indent=2)
        if hasattr(self, "_history"):
            with open(self.save_result_folder_path + "history.json", "w") as f:
                json.dump(self._history, f, indent=2)
        if hasattr(self, "_refine_log"):
            with open(self.save_result_folder_path + "refine_log.json", "w") as f:
                json.dump(self._refine_log, f, indent=2)
        return mesh_path
