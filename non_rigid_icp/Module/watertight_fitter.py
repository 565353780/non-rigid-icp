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
    buildBarycentricSamples,
    sampleDeformedSurface,
)
from non_rigid_icp.Method.nn import NNIndex
from non_rigid_icp.Loss.surface import (
    symmetricChamferLoss,
    pointToPlaneLoss,
    edgeLaplacianLoss,
)
from non_rigid_icp.Metric.chamfer import computeChamferMetrics, computeF1AtThreshold


class WatertightFitter(object):
    """Topology-agnostic non-rigid fitter for very large meshes.

    Source (the watertight mesh) is deformed by a per-vertex displacement field
    so that its surface snaps onto the target (the original mesh). All data
    terms come from surface samples, so the source and target need not share any
    topology or vertex indexing.

    Pipeline:
      1. normalize source/target into the target bbox frame (shared scale)
      2. rigid ICP init (source -> target)
      3. coarse-to-fine optimization of the displacement field with
         symmetric chamfer + edge-laplacian smoothness, with an optional
         point-to-plane refinement stage
      4. de-normalize and report Chamfer / F1 metrics
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
        self.seed = seed

        # mask threshold schedule (in normalized frame). Defaults to a coarse
        # -> fine anneal expressed as multiples of tau_norm, filled in fit().
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
        # sample the target surface for a robust ICP target point cloud
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

    def fit(self) -> o3d.geometry.TriangleMesh:
        """Per-vertex point-to-point non-rigid ICP with fast spatial NN.

        Each outer iteration: (1) fix a per-vertex correspondence from the
        current deformed vertices to the nearest target surface point using a
        spatial NN index (O(N log M), tractable at full resolution), masking out
        unreliable far matches; (2) take a few gradient steps minimizing the
        masked point-to-point data term plus an edge-Laplacian smoothness on the
        displacement field. The mask distance and Laplacian weight are annealed
        coarse-to-fine. The deformation is metric-guarded: the result is only
        kept if it does not worsen the L1 chamfer over the rigid-init baseline.
        """
        assert self.source_mesh is not None and self.target_mesh is not None

        self._rigidInit()

        dev = self.device
        V = np.asarray(self.source_mesh.vertices, dtype=np.float32)
        F = np.asarray(self.source_mesh.triangles)
        self._baseline_vertices = V.copy()

        verts = torch.tensor(V, device=dev, dtype=torch.float32)
        tris = torch.tensor(F, device=dev, dtype=torch.long)

        # vectorized unique edges (no python loop -> works for tens of millions)
        e = torch.cat([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]], dim=0)
        e = torch.sort(e, dim=1).values
        edges = torch.unique(e, dim=0)
        print(
            "[INFO][WatertightFitter::fit] V=",
            V.shape[0],
            "F=",
            F.shape[0],
            "E=",
            edges.shape[0],
        )

        # dense stable target point set (+normals) for correspondence
        target_pts_np, target_nrm_np = sampleMeshSurface(
            self.target_mesh,
            max(self.train_target_samples, 2000000),
            seed=self.seed,
            with_normals=True,
        )
        target_pts = torch.tensor(target_pts_np, device=dev, dtype=torch.float32)
        target_nrm = torch.tensor(target_nrm_np, device=dev, dtype=torch.float32)
        target_index = NNIndex(target_pts_np, device=dev)

        disp = torch.zeros_like(verts, requires_grad=True)
        optimizer = torch.optim.AdamW([disp], lr=self.lr, amsgrad=True)

        tau_norm = (self.L / 2048.0) * self.norm_scale
        if self.mask_dist_schedule is None:
            self.mask_dist_schedule = [m * tau_norm for m in (16.0, 8.0, 4.0, 2.0, 1.0)]
        if self.laplacian_schedule is None:
            self.laplacian_schedule = [
                self.laplacian_weight * f for f in (1.0, 1.0, 0.6, 0.3, 0.1)
            ]
        if self.point_to_plane_schedule is None:
            # ramp up point-to-plane towards the fine refinement stages
            self.point_to_plane_schedule = [
                self.point_to_plane_weight * f for f in (0.0, 0.25, 0.5, 1.0, 2.0)
            ]

        n_stages = len(self.mask_dist_schedule)
        stage_milestones = np.linspace(0, self.outer_iter, n_stages + 1).astype(int)

        loop = trange(self.outer_iter)
        history = []
        for i in loop:
            stage = int(np.searchsorted(stage_milestones, i, side="right") - 1)
            stage = max(0, min(stage, n_stages - 1))
            mask_dist = self.mask_dist_schedule[stage]
            lap_w = self.laplacian_schedule[stage]
            p2p_w = self.point_to_plane_schedule[stage]

            # refresh per-vertex correspondence only periodically: querying all
            # ~14M vertices against the target index is the dominant cost, while
            # the displacement between refreshes is tiny.
            if i == 0 or (i % self.corr_refresh_every == 0):
                with torch.no_grad():
                    cur = (verts + disp).detach()
                    idx, d2 = target_index.query(cur, k=1)
                    idx_t = torch.from_numpy(idx).to(dev)
                    matched = target_pts[idx_t]
                    matched_n = target_nrm[idx_t]
                    weight = (
                        torch.from_numpy(d2).to(dev) < (mask_dist ** 2)
                    ).float().unsqueeze(1)

            n_inner = self.inner_iter if i > 0 else 4
            for _ in range(n_inner):
                optimizer.zero_grad()
                d = verts + disp
                diff = d - matched
                if p2p_w > 0:
                    plane = (diff * matched_n).sum(dim=1, keepdim=True)
                    data = (
                        self.fit_weight * (weight * diff ** 2).sum(dim=1).mean()
                        + p2p_w * (weight * plane ** 2).mean()
                    )
                else:
                    data = self.fit_weight * (weight * diff ** 2).sum(dim=1).mean()
                lap_loss = edgeLaplacianLoss(disp, edges)
                loss = data + lap_w * lap_loss
                loss.backward()
                optimizer.step()

            loop.set_postfix(
                stage=stage,
                data=round(data.item(), 8),
                lap=round(lap_loss.item(), 8),
                matched=round(weight.mean().item(), 3),
            )
            history.append(
                {
                    "iter": i,
                    "stage": stage,
                    "data": data.item(),
                    "lap": lap_loss.item(),
                    "matched_frac": weight.mean().item(),
                    "mask_dist": mask_dist,
                    "lap_w": lap_w,
                    "p2p_w": p2p_w,
                }
            )

        with torch.no_grad():
            deformed = (verts + disp).detach().cpu().numpy().astype(np.float64)
        self.source_mesh.vertices = deformed

        self._history = history
        return self.source_mesh

    def evaluate(self) -> dict:
        """Evaluate the fitted source against the target in the ORIGINAL target
        coordinate frame, using Chamfer and F1@(L/2048)."""
        # de-normalize copies for metric in original coords
        src = self.source_mesh.clone()
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
            **chamfer,
            **f1,
        }
        return metrics

    def fitAndEvaluate(self) -> dict:
        """Fit, then evaluate both the rigid-init baseline and the deformed
        result. Keep the deformed mesh only if it improves the L1 chamfer; this
        guards against the deformation degrading an already-good rigid fit.

        Returns a dict with 'baseline', 'fitted', 'kept' ('fitted'|'baseline')
        and the final 'metrics'.
        """
        self.fit()

        fitted_vertices = np.asarray(self.source_mesh.vertices).copy()
        fitted_metrics = self.evaluate()

        # baseline = rigid-init only (vertices captured at fit start)
        self.source_mesh.vertices = self._baseline_vertices.astype(np.float64)
        baseline_metrics = self.evaluate()

        if fitted_metrics["chamfer_l1"] <= baseline_metrics["chamfer_l1"]:
            self.source_mesh.vertices = fitted_vertices
            kept = "fitted"
            final = fitted_metrics
        else:
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
        }

    def saveResult(self, metrics: Union[dict, None] = None, config: Union[dict, None] = None) -> str:
        if self.save_result_folder_path is None:
            return ""
        # save de-normalized mesh in original target coordinates
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
        return mesh_path
