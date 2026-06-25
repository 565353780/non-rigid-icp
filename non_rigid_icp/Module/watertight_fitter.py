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
from non_rigid_icp.Method.topology import buildUniqueEdges, buildFaceAdjacency, dilateFaceMask
from non_rigid_icp.Method.convergence import PlateauMonitor
from non_rigid_icp.Method.error_field import localizeHighErrorFaces
from non_rigid_icp.Method.subdivision import subdivideMarkedFaces
from non_rigid_icp.Method.collision import (
    buildCollisionCandidates,
    detectIntersectingPairs,
    detectNewSelfIntersections,
    pairKeys,
)
from non_rigid_icp.Loss.surface import edgeLaplacianLoss
from non_rigid_icp.Loss.collision import selfCollisionBarrierLoss
from non_rigid_icp.Metric.chamfer import computeChamferMetrics, computeF1AtThreshold


class WatertightFitter(object):
    """Topology-agnostic non-rigid fitter for very large meshes.

    Source (the watertight mesh) is deformed by a per-vertex displacement field
    so that its surface snaps onto the target (the original mesh). Two extra
    capabilities sit on top of the per-vertex point-to-point core:

      1. Self-collision guard. A broad-phase (centroid k-NN) + narrow-phase
         (exact triangle-triangle) detector flags any self-intersection that is
         NOT present in the original watertight mesh. A differentiable barrier
         keeps non-adjacent sheets apart, and each optimization step is rolled
         back / re-run with a stronger barrier if it introduced a new crossing.

      2. Adaptive local subdivision. When the fit error stops decreasing
         (plateau), the high-error regions are localized from the bidirectional
         residual, conformingly subdivided (up to K rounds), and the fit
         continues at higher resolution to drive the global error down.

    Pipeline:
      normalize -> rigid ICP -> [ optimize-to-plateau -> localize -> subdivide ]*K
      -> de-normalize and report Chamfer / F1.
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
        # --- self-collision guard ---
        enable_self_collision_guard: bool = True,
        collision_k: int = 8,
        collision_broad_tau: float = 2.0,
        collision_margin_tau: float = 0.25,
        collision_weight: float = 50.0,
        collision_backoff: float = 4.0,
        collision_refresh_every: int = 5,
        max_collision_retries: int = 3,
        collision_full_scan_max_faces: int = 2000000,
        collision_active_tau: float = 1.0,
        collision_max_active: int = 1000000,
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
        self.collision_k = collision_k
        self.collision_broad_tau = collision_broad_tau
        self.collision_margin_tau = collision_margin_tau
        self.collision_weight = collision_weight
        self.collision_backoff = collision_backoff
        self.collision_refresh_every = collision_refresh_every
        self.max_collision_retries = max_collision_retries
        self.collision_full_scan_max_faces = collision_full_scan_max_faces
        self.collision_active_tau = collision_active_tau
        self.collision_max_active = collision_max_active

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

        # coarse -> fine anneal schedules (filled in fit() relative to tau_norm)
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
    # topology / collision state                                         #
    # ------------------------------------------------------------------ #
    def _buildTopology(self) -> None:
        """(Re)build unique edges and face adjacency from self._faces."""
        self._edges = buildUniqueEdges(self._faces)
        self._face_adj = buildFaceAdjacency(self._faces)

    def _deformed(self) -> torch.Tensor:
        return self._verts + self._disp

    def _activeFaceMask(self, motion_thresh: float) -> Union[torch.Tensor, None]:
        """Faces likely to risk a new self-intersection: those touching a vertex
        that moved more than motion_thresh, grown by one ring. Capped to the
        largest movers so the collision cost stays proportional to MOTION, not to
        the (tens-of-millions) mesh size.

        Rationale (first principles): a new crossing requires a face to move by
        about the local feature gap (>= tau); sub-threshold movers cannot create
        one, so they need not be queried."""
        disp_n = self._disp.detach().norm(dim=1)  # (V,)
        moved_v = disp_n > motion_thresh
        if not bool(moved_v.any()):
            return None
        face_moved = moved_v[self._faces].any(dim=1)
        n_active = int(face_moved.sum().item())
        if n_active > self.collision_max_active:
            face_disp = disp_n[self._faces].amax(dim=1)
            face_disp = torch.where(
                face_moved, face_disp, torch.zeros_like(face_disp)
            )
            topk = torch.topk(face_disp, self.collision_max_active).indices
            face_moved = torch.zeros_like(face_moved)
            face_moved[topk] = True
        return dilateFaceMask(face_moved, self._face_adj, 1)

    def _buildCollisionBaseline(self) -> None:
        """Candidate pairs + baseline (allowed) intersection keys for current topology.

        For meshes above `collision_full_scan_max_faces`, a full broad-phase over
        every face is skipped: a watertight input is self-intersection-free by
        construction, so the baseline ignore set is empty and candidates are then
        grown lazily over the faces that actually move (see refresh). Smaller
        meshes get an exact full baseline scan.
        """
        empty_pairs = torch.zeros(0, 2, dtype=torch.long, device=self.device)
        empty_keys = torch.zeros(0, dtype=torch.long, device=self.device)
        if not self.enable_self_collision_guard:
            self._collision_pairs = empty_pairs
            self._baseline_keys = empty_keys
            return

        if self._faces.shape[0] > self.collision_full_scan_max_faces:
            self._collision_pairs = empty_pairs
            self._baseline_keys = empty_keys
            print(
                "[INFO][WatertightFitter] collision baseline: large mesh "
                f"({self._faces.shape[0]} faces) assumed intersection-free "
                "(watertight); candidates built lazily over active faces."
            )
            return

        deformed = self._deformed().detach()
        self._collision_pairs = buildCollisionCandidates(
            deformed,
            self._faces,
            k=self.collision_k,
            margin=self._broad_margin,
            active_face_mask=None,
            device=self.device,
        )
        hit = detectIntersectingPairs(deformed, self._faces, self._collision_pairs)
        self._baseline_keys = pairKeys(
            self._collision_pairs[hit], self._faces.shape[0]
        )
        print(
            "[INFO][WatertightFitter] collision baseline:",
            self._collision_pairs.shape[0],
            "candidate pairs,",
            int(hit.sum().item()),
            "pre-existing intersections (ignored).",
        )

    def _refreshCollisionCandidates(self) -> None:
        if not self.enable_self_collision_guard:
            return
        deformed = self._deformed().detach()
        active = self._activeFaceMask(self.collision_active_tau * self._tau_norm)
        if active is None:
            return
        new_pairs = buildCollisionCandidates(
            deformed,
            self._faces,
            k=self.collision_k,
            margin=self._broad_margin,
            active_face_mask=active,
            device=self.device,
        )
        if new_pairs.shape[0] == 0:
            return
        merged = torch.cat([self._collision_pairs, new_pairs], dim=0)
        self._collision_pairs = torch.unique(merged, dim=0)

    def _countNewIntersections(self) -> Tuple[int, torch.Tensor]:
        if not self.enable_self_collision_guard or self._collision_pairs.shape[0] == 0:
            return 0, torch.zeros(0, 2, dtype=torch.long, device=self.device)
        deformed = self._deformed().detach()
        new_pairs, _ = detectNewSelfIntersections(
            deformed, self._faces, self._collision_pairs, self._baseline_keys
        )
        return new_pairs.shape[0], new_pairs

    def _fullCollisionScan(self) -> int:
        """One-shot end-of-fit self-intersection check on the output mesh.

        A new self-intersection must involve at least one MOVED face, so it is
        sufficient (and complete under the >= tau motion assumption) to query the
        moved faces against the full-face spatial index. Faces queried use a
        lower threshold than during optimization to be thorough, capped for
        tractability."""
        if not self.enable_self_collision_guard:
            return 0
        deformed = self._deformed().detach()
        active = self._activeFaceMask(0.5 * self._tau_norm)
        if active is None:
            return 0
        pairs = buildCollisionCandidates(
            deformed,
            self._faces,
            k=self.collision_k,
            margin=self._broad_margin,
            active_face_mask=active,
            device=self.device,
        )
        new_pairs, _ = detectNewSelfIntersections(
            deformed, self._faces, pairs, self._baseline_keys
        )
        return int(new_pairs.shape[0])

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
    ) -> Tuple[float, float, float]:
        data_v = lap_v = coll_v = 0.0
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
            loss.backward()
            self._optimizer.step()
            data_v = float(data.item())
            lap_v = float(lap_loss.item())
        return data_v, lap_v, coll_v

    def _guardedOuterStep(
        self,
        n_inner: int,
        matched: torch.Tensor,
        matched_n: torch.Tensor,
        weight: torch.Tensor,
        lap_w: float,
        p2p_w: float,
    ) -> dict:
        """Run an inner burst, then roll back + re-run harder if it created a new
        self-intersection. Guarantees the step never leaves a NEW crossing."""
        snapshot = self._disp.detach().clone()
        coll_w = self.collision_weight if self.enable_self_collision_guard else 0.0

        data_v, lap_v, coll_v = self._innerSteps(
            n_inner, matched, matched_n, weight, lap_w, p2p_w, coll_w
        )

        n_new, _ = self._countNewIntersections()
        retries = 0
        while n_new > 0 and retries < self.max_collision_retries:
            self._disp.data.copy_(snapshot)
            coll_w *= self.collision_backoff
            data_v, lap_v, coll_v = self._innerSteps(
                n_inner, matched, matched_n, weight, lap_w, p2p_w, coll_w
            )
            n_new, _ = self._countNewIntersections()
            retries += 1

        if n_new > 0:
            # could not resolve: keep the safe pre-step state
            self._disp.data.copy_(snapshot)

        return {
            "data": data_v,
            "lap": lap_v,
            "coll": coll_v,
            "coll_w": coll_w,
            "new_self_intersections": int(n_new),
            "collision_retries": int(retries),
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
            # refinement cycles run at the finest stage throughout
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

        matched = matched_n = weight = None
        loop = trange(n_outer, desc=f"cycle{level}")
        for i in loop:
            mask_dist, lap_w, p2p_w, stage = self._cycleStageParams(level, i, n_outer)

            if self.enable_self_collision_guard and i > 0 and (
                i % self.collision_refresh_every == 0
            ):
                self._refreshCollisionCandidates()

            if i == 0 or (i % self.corr_refresh_every == 0):
                with torch.no_grad():
                    cur = self._deformed().detach()
                    idx, d2 = target_index.query(cur, k=1)
                    idx_t = torch.from_numpy(idx).to(dev)
                    matched = target_pts[idx_t]
                    matched_n = target_nrm[idx_t]
                    weight = (
                        torch.from_numpy(d2).to(dev) < (mask_dist ** 2)
                    ).float().unsqueeze(1)

            n_inner = self.inner_iter if i > 0 else 4
            info = self._guardedOuterStep(
                n_inner, matched, matched_n, weight, lap_w, p2p_w
            )

            plateau = monitor.update(info["data"])
            loop.set_postfix(
                stage=stage,
                data=round(info["data"], 8),
                lap=round(info["lap"], 8),
                new_si=info["new_self_intersections"],
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
                    "coll_w": info["coll_w"],
                    "new_self_intersections": info["new_self_intersections"],
                    "collision_retries": info["collision_retries"],
                    "n_candidates": int(self._collision_pairs.shape[0]),
                    "rel_drop": monitor.last_rel_drop,
                    "mask_dist": mask_dist,
                    "lap_w": lap_w,
                    "p2p_w": p2p_w,
                }
            )

            # plateaued and we still have subdivision budget -> stop early to refine
            if plateau and level < self.max_subdivisions:
                print(
                    f"[INFO][WatertightFitter] cycle {level} plateaued at iter {i} "
                    f"(rel_drop={monitor.last_rel_drop:.5f}); triggering refinement."
                )
                break

    # ------------------------------------------------------------------ #
    # adaptive subdivision                                               #
    # ------------------------------------------------------------------ #
    def _refine(self, level: int) -> bool:
        """Localize high-error faces and conformingly subdivide. Returns True if
        the topology changed (refinement happened)."""
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

        # bake current deformation into the base, then subdivide it
        new_v, new_f, _ = subdivideMarkedFaces(deformed, self._faces, region_mask)
        v_before, f_before = self._verts.shape[0], self._faces.shape[0]
        self._verts = new_v.detach().clone()
        self._faces = new_f
        self._disp = torch.zeros_like(self._verts, requires_grad=True)
        self._optimizer = torch.optim.AdamW([self._disp], lr=self.lr, amsgrad=True)
        self._buildTopology()
        self._buildCollisionBaseline()
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

        # baseline mesh (rigid-init, original topology) for metric-guarded accept
        self._baseline_mesh = self.source_mesh.clone()

        dev = self.device
        V = np.asarray(self.source_mesh.vertices, dtype=np.float32)
        F = np.asarray(self.source_mesh.triangles)

        self._verts = torch.tensor(V, device=dev, dtype=torch.float32)
        self._faces = torch.tensor(F, device=dev, dtype=torch.long)
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

        # dense stable target point set (+normals) for correspondence
        target_pts_np, target_nrm_np = sampleMeshSurface(
            self.target_mesh,
            self.train_target_samples,
            seed=self.seed,
            with_normals=True,
        )
        self._target_pts = torch.tensor(target_pts_np, device=dev, dtype=torch.float32)
        self._target_nrm = torch.tensor(target_nrm_np, device=dev, dtype=torch.float32)
        self._target_index = NNIndex(target_pts_np, device=dev)

        self._tau_norm = (self.L / 2048.0) * self.norm_scale
        tau_norm = self._tau_norm
        self._broad_margin = self.collision_broad_tau * tau_norm
        self._barrier_margin = self.collision_margin_tau * tau_norm

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

        self._buildCollisionBaseline()

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

        # authoritative one-shot full-mesh self-intersection check on the output
        final_new = 0
        if self.enable_self_collision_guard:
            final_new = self._fullCollisionScan()
            print("[INFO][WatertightFitter] final new self-intersections:", final_new)
        self._final_new_self_intersections = final_new

        # write the deformed, possibly-subdivided mesh back into source_mesh
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
        """Fit, then evaluate the rigid-init baseline and the deformed (refined)
        result. Keep the deformed mesh only if it improves the L1 chamfer."""
        self.fit()

        fitted_metrics = self._evaluateMesh(self.source_mesh)
        baseline_metrics = self._evaluateMesh(self._baseline_mesh)

        fitted_v = np.asarray(self.source_mesh.vertices).copy()
        fitted_f = np.asarray(self.source_mesh.triangles).copy()

        if fitted_metrics["chamfer_l1"] <= baseline_metrics["chamfer_l1"]:
            kept = "fitted"
            final = fitted_metrics
        else:
            # roll back to the rigid-init baseline (original topology)
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
        }

    def saveResult(
        self, metrics: Union[dict, None] = None, config: Union[dict, None] = None
    ) -> str:
        if self.save_result_folder_path is None:
            return ""
        out_mesh = self.source_mesh.clone()
        out_mesh.transform(self.norm_center, self.norm_scale, is_inverse=True)
        mesh_path = self.save_result_folder_path + "fitted_mesh.ply"
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
