"""Implicit-field projection fitter (no optimization / no IPC / no ARAP).

The watertight source is already very close to the original target, so instead of
a barrier-regularized gradient optimization we directly PROJECT each source vertex
onto the target's implicit surface (winding-number SDF) by closest-point steps:

    v' = v - phi(v) * grad phi(v)        (capped to ||v' - v|| <= step)

Self-intersection is prevented from FIRST PRINCIPLES, not by soft barriers: the
straight segment from a vertex's frozen rest position on the watertight mesh
(`ref`) to its proposed position must not pierce any non-incident face of the
current mesh (`Method/trajectory_guard`). A crossing means the vertex tunnelled
through a sheet; the step is then bisected back to the largest safe fraction.
Subdivision carries `ref` (midpoint = the interpolated point on the watertight
mesh), so the invariant survives refinement. A final authoritative
`findSelfIntersections` scan + local pull-back guarantees zero new crossings in
the output.

Pipeline:
  normalize -> (rigid ICP) -> build implicit field ->
  [ project-to-plateau (guarded) -> localize -> subdivide ]*K ->
  authoritative final gate -> de-normalize and report.

Public API mirrors `Module/watertight_fitter.WatertightFitter` so the demo /
evaluation harness is shared.
"""

import os
import json
import torch
import numpy as np
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
)
from non_rigid_icp.Method.convergence import PlateauMonitor
from non_rigid_icp.Method.error_field import localizeHighErrorFaces
from non_rigid_icp.Method.subdivision import subdivideMarkedFaces
from non_rigid_icp.Method.self_intersection import findSelfIntersections
from non_rigid_icp.Method.collision import pairKeys
from non_rigid_icp.Method.implicit_field import ImplicitField
from non_rigid_icp.Method.thickness import (
    vertexWallPartner,
    inwardComponentCap,
)
from non_rigid_icp.Method.trajectory_guard import largestSafeStep
from non_rigid_icp.Metric.chamfer import computeChamferMetrics, computeF1AtThreshold


class ProjectionFitter(object):
    def __init__(
        self,
        device: str = "cuda",
        # --- projection sweeps ---
        step_tau: float = 1.0,
        max_sweeps: int = 24,
        min_move_tau: float = 0.02,
        sweep_plateau_window: int = 3,
        sweep_plateau_rel_tol: float = 5e-2,
        sweep_plateau_patience: int = 2,
        # --- thin-shell aware caps ---
        gap_margin_tau: float = 0.5,
        max_thickness_tau: float = 30.0,
        thickness_frac: float = 0.5,
        # --- trajectory self-intersection guard ---
        enable_guard: bool = True,
        guard_inflate_tau: float = 0.0,
        guard_iters: int = 3,
        bisect_steps: int = 7,
        # --- optional tangential smoothing of the displacement field ---
        smooth_lambda: float = 0.0,
        smooth_iters: int = 0,
        # --- final acceptance ---
        strict_no_intersection: bool = True,
        final_resolve_rounds: int = 40,
        resolve_dilation_rings: int = 2,
        # --- rigid init ---
        rigid_init: bool = True,
        # --- adaptive subdivision (reused atoms) ---
        max_subdivisions: int = 4,
        error_mult: float = 2.0,
        error_quantile: float = 0.9,
        max_refine_faces: Union[int, None] = 1500000,
        min_component_faces: int = 4,
        dilation_rings: int = 1,
        # --- sampling / eval ---
        train_target_samples: int = 2000000,
        eval_samples: int = 2000000,
        save_result_folder_path: Union[str, None] = "auto",
        seed: int = 0,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.step_tau = step_tau
        self.max_sweeps = max_sweeps
        self.min_move_tau = min_move_tau
        self.sweep_plateau_window = sweep_plateau_window
        self.sweep_plateau_rel_tol = sweep_plateau_rel_tol
        self.sweep_plateau_patience = sweep_plateau_patience

        self.gap_margin_tau = gap_margin_tau
        self.max_thickness_tau = max_thickness_tau
        self.thickness_frac = thickness_frac

        self.enable_guard = enable_guard
        self.guard_inflate_tau = guard_inflate_tau
        self.guard_iters = guard_iters
        self.bisect_steps = bisect_steps

        self.smooth_lambda = smooth_lambda
        self.smooth_iters = smooth_iters

        self.strict_no_intersection = strict_no_intersection
        self.final_resolve_rounds = final_resolve_rounds
        self.resolve_dilation_rings = resolve_dilation_rings

        self.rigid_init = rigid_init

        self.max_subdivisions = max_subdivisions
        self.error_mult = error_mult
        self.error_quantile = error_quantile
        self.max_refine_faces = max_refine_faces
        self.min_component_faces = min_component_faces
        self.dilation_rings = dilation_rings

        self.train_target_samples = train_target_samples
        self.eval_samples = eval_samples
        self.seed = seed

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

    # ------------------------------------------------------------------ #
    # setup                                                              #
    # ------------------------------------------------------------------ #
    def loadMeshes(self, source_mesh: Mesh, target_mesh: Mesh) -> bool:
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.norm_center, self.norm_scale, self.L = normalizePairToTargetFrame(
            source_mesh, target_mesh
        )
        print(
            "[INFO][ProjectionFitter::loadMeshes] L(target max bbox edge)=",
            round(self.L, 6),
            "norm_scale=",
            round(self.norm_scale, 6),
        )
        return True

    def _rigidInit(self) -> None:
        if not self.rigid_init:
            return
        source_o3d = self.source_mesh.toO3DMesh()
        target_pts = sampleMeshSurface(
            self.target_mesh, min(self.train_target_samples, 200000), seed=self.seed
        )
        target_pcd = toPointCloud(target_pts)
        transformation = icp(source_o3d, target_pcd)
        if transformation is None:
            print("[WARN][ProjectionFitter::_rigidInit] ICP failed, skip rigid init")
            return
        source_o3d.transform(transformation)
        self.source_mesh.vertices = np.asarray(source_o3d.vertices)
        return

    def _buildTopology(self) -> None:
        self._edges = buildUniqueEdges(self._faces)
        self._face_adj = buildFaceAdjacency(self._faces)

    # ------------------------------------------------------------------ #
    # authoritative self-intersection                                    #
    # ------------------------------------------------------------------ #
    def _buildBaseline(self) -> None:
        """Pre-existing (allowed) intersections, scanned on the CLEAN reference.

        Scanning `_ref_verts` (carried through subdivision) records only the
        source's genuine pre-existing intersections, so every crossing the fit
        creates is fought rather than silently legitimized."""
        inter = findSelfIntersections(
            self._ref_verts.detach(), self._faces, inflate=0.0, exclude_ring=1
        )
        self._baseline_keys = pairKeys(inter, self._faces.shape[0])
        print(
            f"[INFO][ProjectionFitter] baseline scan (clean ref): {inter.shape[0]} "
            "pre-existing intersections (ignored)."
        )

    def _newIntersections(self) -> Tuple[int, torch.Tensor]:
        """Full-mesh authoritative count of NEW (non-baseline) crossings."""
        inter = findSelfIntersections(
            self._verts.detach(), self._faces, inflate=0.0, exclude_ring=1
        )
        if inter.shape[0] == 0:
            return 0, inter
        if self._baseline_keys.numel() > 0:
            keys = pairKeys(inter, self._faces.shape[0])
            inter = inter[~torch.isin(keys, self._baseline_keys)]
        return int(inter.shape[0]), inter

    # ------------------------------------------------------------------ #
    # projection                                                         #
    # ------------------------------------------------------------------ #
    def _smooth(self, verts: torch.Tensor) -> torch.Tensor:
        """Optional uniform-Laplacian smoothing of the displacement field."""
        if self.smooth_lambda <= 0.0 or self.smooth_iters <= 0:
            return verts
        e0, e1 = self._edges[:, 0], self._edges[:, 1]
        ones = torch.ones(e0.shape[0], device=verts.device)
        d = verts - self._ref_verts
        for _ in range(self.smooth_iters):
            nb_sum = torch.zeros_like(d)
            nb_sum.index_add_(0, e0, d[e1])
            nb_sum.index_add_(0, e1, d[e0])
            cnt = torch.zeros(verts.shape[0], device=verts.device)
            cnt.index_add_(0, e0, ones)
            cnt.index_add_(0, e1, ones)
            d_avg = nb_sum / cnt.clamp(min=1.0).unsqueeze(1)
            d = (1.0 - self.smooth_lambda) * d + self.smooth_lambda * d_avg
        return self._ref_verts + d

    def _projectCycle(self, level: int) -> None:
        tau = self._tau_norm
        min_move = self.min_move_tau * tau
        guard_inflate = self.guard_inflate_tau * tau

        # per-vertex wall thickness + unit direction toward the opposite layer
        # (measured on the CLEAN reference so the two layers are still ordered).
        thickness, toward_dir = vertexWallPartner(
            self._ref_verts, self._faces, self.max_thickness_tau * tau
        )
        # Per-VERTEX step cap: keep each update small so a vertex cannot overshoot
        # its surface and the trajectory guard stays a valid linearization. For
        # thin-shell vertices the cap is tightened to a fraction of the LOCAL wall
        # thickness, so a single sweep can never tunnel a 0.16-tau wall (a fixed
        # 1-tau step would jump several wall widths and defeat the guard). Thick /
        # single-layer vertices (thickness == +inf) keep the full step_tau.
        step_cap = torch.full(
            (self._verts.shape[0],), self.step_tau * tau, device=self._verts.device
        )
        finite = torch.isfinite(thickness)
        step_cap[finite] = torch.minimum(
            step_cap[finite], self.thickness_frac * thickness[finite]
        )
        # how far each layer may move toward its partner so that, even if both
        # move their full allowance, the residual gap is still gap_margin.
        allowance = torch.clamp(
            (thickness - self.gap_margin_tau * tau) * 0.5, min=0.0
        )
        n_wall = int(torch.isfinite(thickness).sum().item())
        fin = thickness[torch.isfinite(thickness)]
        med = float(fin.median()) / tau if fin.numel() > 0 else float("nan")
        print(
            f"[INFO][ProjectionFitter] level {level}: {n_wall}/{thickness.numel()} "
            f"wall verts, median wall {med:.2f}tau."
        )

        monitor = PlateauMonitor(
            window=self.sweep_plateau_window,
            rel_tol=self.sweep_plateau_rel_tol,
            patience=self.sweep_plateau_patience,
            min_updates=2 * self.sweep_plateau_window,
        )

        loop = trange(self.max_sweeps, desc=f"proj{level}")
        for sweep in loop:
            proposed, full = self._field.project(self._verts, step_cap)
            # thin-shell collapse guard: cap cumulative motion toward the wall
            # partner so opposing layers can never close past gap_margin (only
            # resists motion that CLOSES the wall; translation/slide are free).
            proposed = inwardComponentCap(
                proposed, self._ref_verts, toward_dir, allowance
            )

            n_clamp = 0
            if self.enable_guard:
                # Guard each mover's trajectory [ref, pos] against the PROPOSED
                # (post-move) configuration, iterated to a fixpoint. Testing the
                # pre-move mesh misses the dominant double-layer collapse: when
                # BOTH opposing layers step toward each other simultaneously
                # (Jacobi), each individual segment is shorter than the wall gap
                # so neither pierces a pre-move face -- yet together they cross.
                # In the proposed config the moved opposite face lies on the
                # mover's trajectory, so the crossing IS detected and bisected
                # back. Running every sweep keeps incremental crossings from ever
                # accumulating, so the candidate set stays tiny.
                cur = proposed
                for _g in range(self.guard_iters):
                    move_now = (cur - self._verts).norm(dim=1)
                    idx = torch.nonzero(move_now > min_move, as_tuple=False).reshape(-1)
                    if idx.numel() == 0:
                        break
                    safe_pos, clamped = largestSafeStep(
                        self._ref_verts[idx],
                        cur[idx],
                        cur,
                        self._faces,
                        owner_vid=idx,
                        inflate=guard_inflate,
                        n_bisect=self.bisect_steps,
                    )
                    nc = int(clamped.sum().item())
                    n_clamp += nc
                    if nc == 0:
                        break
                    cur = cur.clone()
                    cur[idx] = safe_pos
                proposed = cur

            proposed = self._smooth(proposed)
            move = (proposed - self._verts).norm(dim=1)
            self._verts = proposed
            mean_move = float(move.mean().item())
            max_move = float(move.max().item())
            n_moving = int((move > min_move).sum().item())
            # convergence is the fitting RESIDUAL (distance to the target surface)
            # flattening -- NOT the per-sweep move, which stays constant while a
            # vertex advances at the capped rate and would be misread as a plateau.
            mean_residual = float(full.mean().item())

            plateau = monitor.update(mean_residual)
            loop.set_postfix(
                mean_move=round(mean_move / tau, 4),
                max_move=round(max_move / tau, 4),
                moving=n_moving,
                clamped=n_clamp,
                resid=round(float(full.mean().item()) / tau, 4),
            )
            self._history.append(
                {
                    "level": level,
                    "sweep": sweep,
                    "mean_move_tau": mean_move / tau,
                    "max_move_tau": max_move / tau,
                    "n_moving": n_moving,
                    "n_clamped": n_clamp,
                    "mean_residual_tau": float(full.mean().item()) / tau,
                }
            )
            if plateau or n_moving == 0:
                break

    # ------------------------------------------------------------------ #
    # adaptive subdivision (reused atoms)                                #
    # ------------------------------------------------------------------ #
    def _refine(self, level: int) -> bool:
        region_mask, _, stats = localizeHighErrorFaces(
            self._verts.detach(),
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
            f"[INFO][ProjectionFitter] refine level {level}: {n_region} faces "
            f"marked (thr={stats['threshold']:.6f}, "
            f"max_err={stats['face_error_max']:.6f})."
        )
        if n_region == 0:
            return False

        v_before, f_before = self._verts.shape[0], self._faces.shape[0]
        new_v, new_f, _, extra = subdivideMarkedFaces(
            self._verts.detach(),
            self._faces,
            region_mask,
            extra_vertex_attrs=[self._ref_verts],
        )
        self._verts = new_v.detach().clone()
        self._faces = new_f
        self._ref_verts = extra[0].detach().clone()
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
            f"[INFO][ProjectionFitter] subdivided: V {v_before}->{self._verts.shape[0]}, "
            f"F {f_before}->{self._faces.shape[0]}."
        )
        return True

    # ------------------------------------------------------------------ #
    # final hard gate                                                    #
    # ------------------------------------------------------------------ #
    def _finalGate(self) -> int:
        """Guarantee zero NEW self-intersections by PINNING offending regions to
        the clean reference until the authoritative scan is clean.

        First principles for a PROVABLE guarantee: the clean reference
        (`_ref_verts`, carried through subdivision) is intersection-free modulo
        the recorded baseline. Each round we take every vertex incident to a NEW
        crossing and pin it EXACTLY to its rest position; the pinned set only
        grows (a vertex never un-pins). In the worst case every vertex is pinned,
        which reproduces the clean rest mesh, so the non-baseline crossing count
        is driven monotonically to zero in a finite number of rounds. (Halving
        toward rest, in contrast, can stall: a region asymptotically approaches
        but never reaches the clean config, so a residual crossing survives.)
        """
        n_new, inter = self._newIntersections()
        print(f"[INFO][ProjectionFitter] final gate: {n_new} new crossings.")
        if n_new == 0 or not self.strict_no_intersection:
            return n_new
        e0, e1 = self._edges[:, 0], self._edges[:, 1]
        pinned = torch.zeros(
            self._verts.shape[0], dtype=torch.bool, device=self._verts.device
        )
        for r in range(self.final_resolve_rounds):
            involved_faces = torch.unique(inter.reshape(-1))
            vids = torch.unique(self._faces[involved_faces].reshape(-1))
            fresh = torch.zeros_like(pinned)
            fresh[vids] = True
            # Dilate the freshly dirty set by a few vertex rings before pinning.
            # A surviving crossing is always a rest-face vs projected-face pair on
            # the boundary between the pinned (rest) region and the still-moved
            # region; pinning ONLY the two faces advances that boundary just one
            # ring per round, which crawls across extended thin sheets. Growing a
            # rest BUFFER around each crossing turns those boundary pairs into
            # rest-vs-rest (baseline, not new) and collapses the dirty annulus
            # geometrically, so the scan reaches exactly zero in a few rounds.
            for _ in range(self.resolve_dilation_rings):
                acc = fresh.to(torch.int8)
                acc.index_add_(0, e0, fresh[e1].to(torch.int8))
                acc.index_add_(0, e1, fresh[e0].to(torch.int8))
                fresh = acc > 0
            pinned |= fresh
            # snap the whole accumulated dirty set back to the clean reference
            self._verts[pinned] = self._ref_verts[pinned]
            n_new, inter = self._newIntersections()
            print(
                f"[INFO][ProjectionFitter] resolve round {r}: {n_new} new "
                f"crossings remain ({int(pinned.sum().item())} verts pinned to "
                "rest)."
            )
            if n_new == 0:
                break
        return n_new

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
        self._ref_verts = self._verts.detach().clone()
        self._buildTopology()
        print(
            "[INFO][ProjectionFitter::fit] V=", self._verts.shape[0],
            "F=", self._faces.shape[0], "E=", self._edges.shape[0],
        )

        # implicit field of the (normalized) target
        tV = np.asarray(self.target_mesh.vertices, dtype=np.float32)
        tF = np.asarray(self.target_mesh.triangles)
        self._field = ImplicitField(tV, tF, device=dev)

        # target samples for error localization (reused atom)
        target_pts_np = sampleMeshSurface(
            self.target_mesh, self.train_target_samples, seed=self.seed
        )
        self._target_pts = torch.tensor(target_pts_np, device=dev, dtype=torch.float32)
        self._target_index = NNIndex(target_pts_np, device=dev)

        # tau in the normalized frame: (L/2048)*scale = 0.9/2048
        self._tau_norm = (self.L / 2048.0) * self.norm_scale

        self._history = []
        self._refine_log = []

        self._buildBaseline()

        for level in range(self.max_subdivisions + 1):
            self._projectCycle(level)
            if level < self.max_subdivisions:
                if not self._refine(level):
                    print(
                        "[INFO][ProjectionFitter] no high-error region left; "
                        "stopping subdivision early."
                    )
                    break

        self._final_new_self_intersections = self._finalGate()

        with torch.no_grad():
            deformed = self._verts.detach().cpu().numpy().astype(np.float64)
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
        return {
            "L": self.L,
            "tau": tau,
            "eval_samples": self.eval_samples,
            "n_vertices": int(np.asarray(src_mesh.vertices).shape[0]),
            "n_faces": int(np.asarray(src_mesh.triangles).shape[0]),
            **chamfer,
            **f1,
        }

    def evaluate(self) -> dict:
        return self._evaluateMesh(self.source_mesh)

    def fitAndEvaluate(self) -> dict:
        self.fit()
        fitted_metrics = self._evaluateMesh(self.source_mesh)
        baseline_metrics = self._evaluateMesh(self._baseline_mesh)
        clean = int(getattr(self, "_final_new_self_intersections", 0)) == 0

        if fitted_metrics["chamfer_l1"] <= baseline_metrics["chamfer_l1"]:
            kept = "fitted"
            final = fitted_metrics
        else:
            self.source_mesh.vertices = np.asarray(self._baseline_mesh.vertices).copy()
            self.source_mesh.triangles = np.asarray(
                self._baseline_mesh.triangles
            ).copy()
            self.source_mesh.vertex_colors = None
            self.source_mesh.vertex_normals = None
            self.source_mesh.triangle_normals = None
            kept = "baseline"
            final = baseline_metrics
            print(
                "[INFO][ProjectionFitter] projection did not improve metric; "
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
