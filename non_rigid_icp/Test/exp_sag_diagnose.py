"""Diagnostic: after a single projection step (step_frac=1.0 -> all vertices on
target), measure the per-face SAG distribution on the ROI-cropped case1, in
units of tau (=L/2048). This tells us, from data, how many faces are GENUINELY
tented over a target feature vs. sitting at the tolerance floor -- i.e. the
right sag threshold, and whether subdivision is even warranted.
"""

import numpy as np
import torch

from non_rigid_icp.Demo.watertight_fitter import data_dict
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter
from non_rigid_icp.Method.implicit_field import ImplicitField
from non_rigid_icp.Method.error_field import faceCentroids, faceSagError


ROI_BBOXES = (
    {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
    {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
)


def main():
    src_path, tgt_path = data_dict["watertight_case1"]
    source_mesh = Mesh(src_path)
    target_mesh = Mesh(tgt_path)

    fitter = WatertightFitter(
        device="cuda",
        laplacian_weight=30.0,
        normal_gate=False,
        train_target_samples=2000000,
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        max_subdivisions=0,
        refine_sag_mult=0.0,  # disable auto-refine; we just project once
        eval_bboxes=ROI_BBOXES,
        prefit_crop_bboxes=ROI_BBOXES,
        prefit_crop_mode="centroid",
        save_result_folder_path="./output/sag_diag/",
    )
    fitter.loadMeshes(source_mesh, target_mesh)

    # one projection step (step_frac=1.0): drives every vertex onto the target.
    out = fitter.fitStepwiseClamped(
        n_steps=2, step_frac=1.0, gd_lr=2.0, lap_w=30.0,
        max_subdivisions=0, converge_abs_tau=0.0,
        compute_chamfer=False, crop_eval=False,
    )
    tau = out["tau"] * fitter.norm_scale  # tau in the normalized frame
    dev = fitter.device

    deformed = fitter._deformed().detach()
    faces = fitter._faces
    tV = np.asarray(fitter.target_mesh.vertices, dtype=np.float32)
    tF = np.asarray(fitter.target_mesh.triangles)
    field = ImplicitField(tV, tF, device=dev)

    cp_v, _, _ = field.closestPoints(deformed)
    vdist = (cp_v - deformed).norm(dim=1)
    cents = faceCentroids(deformed, faces)
    cp_c, _, _ = field.closestPoints(cents)
    cdist = (cp_c - cents).norm(dim=1)
    sag = faceSagError(vdist, cdist, faces)

    sag_tau = (sag / tau).cpu().numpy()
    cdist_tau = (cdist / tau).cpu().numpy()
    nF = sag_tau.size

    print(f"\n# SAG DISTRIBUTION (n_faces={nF}, tau={tau:.6e} normalized)")
    print(f"  vertex dist mean = {float(vdist.mean())/tau:.3f} tau  "
          f"(max {float(vdist.max())/tau:.3f} tau)")
    print(f"  centroid dist: mean {cdist_tau.mean():.3f} tau, "
          f"max {cdist_tau.max():.3f} tau")
    print(f"  sag: mean {sag_tau.mean():.4f} tau, "
          f"median {np.median(sag_tau):.4f} tau, max {sag_tau.max():.3f} tau")
    print("\n  faces with sag above K*tau:")
    for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        n = int((sag_tau > k).sum())
        print(f"    sag > {k:>2} tau : {n:>10}  ({100.0*n/nF:.4f}%)")
    print("\n  faces with BOTH sag>K*tau AND centroid_dist>K*tau:")
    for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        n = int(((sag_tau > k) & (cdist_tau > k)).sum())
        print(f"    both > {k:>2} tau : {n:>10}  ({100.0*n/nF:.4f}%)")


if __name__ == "__main__":
    main()
