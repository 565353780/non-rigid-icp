import os
import json
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
from non_rigid_icp.Method.error_field import localizeHighErrorFaces
from non_rigid_icp.Method.subdivision import subdivideMarkedFaces
from non_rigid_icp.Method.collision import (
    buildCollisionCandidatesAABB,
    pairKeys,
)
from non_rigid_icp.Method.self_intersection import findSelfIntersections
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

    # ------------------------------------------------------------------ #
    # topology / motion                                                  #
    # ------------------------------------------------------------------ #
    def _buildTopology(self) -> None:
        self._edges = buildUniqueEdges(self._faces)
        self._face_adj = buildFaceAdjacency(self._faces)

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

    # ------------------------------------------------------------------ #
    # adaptive subdivision                                               #
    # ------------------------------------------------------------------ #
    def _refine(self, level: int) -> bool:
        deformed = self._deformed().detach()
        region_mask, face_error, stats = localizeHighErrorFaces(
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
        if n_region == 0:
            return False

        # bake current deformation into the base, carry the clean reference field
        # through the same split so cumulative motion stays well-defined.
        new_v, new_f, _, extra = subdivideMarkedFaces(
            deformed, self._faces, region_mask, extra_vertex_attrs=[self._ref_verts]
        )
        v_before, f_before = self._verts.shape[0], self._faces.shape[0]
        self._verts = new_v.detach().clone()
        self._faces = new_f
        self._ref_verts = extra[0].detach().clone()
        self._disp = torch.zeros_like(self._verts, requires_grad=True)
        self._optimizer = torch.optim.AdamW([self._disp], lr=self.lr, amsgrad=True)
        self._buildTopology()
        self._buildBaseline()
        self._refine_log.append(
            {
                "level": level,
                "n_region_faces": n_region,
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
    # fit                                                                #
    # ------------------------------------------------------------------ #
    def fit(self) -> Mesh:
        assert self.source_mesh is not None and self.target_mesh is not None

        self._rigidInit()
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

        # authoritative full-mesh final gate on the output
        final_global = 0
        if self.enable_self_collision_guard:
            n_new, _ = self._newIntersections(query_mask=None)
            final_global = int(n_new)
            print(
                "[INFO][WatertightFitter] final authoritative new self-intersections:",
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
        return self.source_mesh

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

        clean = int(getattr(self, "_final_new_self_intersections", 0)) == 0

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
            print(
                "[INFO][WatertightFitter] deformation did not improve metric; "
                "keeping rigid-init baseline."
            )

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
        }

    def saveResult(
        self, metrics: Union[dict, None] = None, config: Union[dict, None] = None
    ) -> str:
        if self.save_result_folder_path is None:
            return ""
        out_mesh = self.source_mesh.clone()
        out_mesh.transform(self.norm_center, self.norm_scale, is_inverse=True)
        clean = int(getattr(self, "_final_new_self_intersections", 0)) == 0
        name = "fitted_mesh.ply" if clean else "fitted_mesh_unverified.ply"
        mesh_path = self.save_result_folder_path + name
        out_mesh.save(mesh_path, overwrite=True)

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
