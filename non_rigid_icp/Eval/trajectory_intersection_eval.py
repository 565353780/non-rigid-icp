"""Trajectory self-intersection evaluation (the user-defined criterion).

Given a fitted mesh and its trajectory reference mesh (same topology, every
vertex at its clean watertight-rest position), this reports how many vertices'
straight trajectory from rest to fitted position pierces a non-incident face of
the fitted mesh. Zero crossings == the fit introduced no trajectory-defined
self-intersection.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Eval.trajectory_intersection_eval \
        --mesh      output/case1_v4_trajectory_guard/fitted_mesh.ply \
        --reference output/case1_v4_trajectory_guard/trajectory_reference_mesh.ply \
        --out       output/case1_v4_trajectory_guard/trajectory_intersection_report.json
"""

import json
import time
import argparse
import numpy as np
import torch

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.trajectory_guard import segmentMeshIntersections


def evaluateTrajectoryFile(
    mesh_path: str,
    reference_path: str,
    device: str = "cuda",
    inflate_tau: float = 0.0,
) -> dict:
    t0 = time.time()
    fitted = Mesh(mesh_path)
    reference = Mesh(reference_path)

    cur = torch.from_numpy(np.asarray(fitted.vertices, dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.asarray(fitted.triangles, dtype=np.int64)).to(device)
    ref = torch.from_numpy(np.asarray(reference.vertices, dtype=np.float32)).to(device)
    ref_faces = np.asarray(reference.triangles, dtype=np.int64)
    load_s = time.time() - t0

    if cur.shape[0] != ref.shape[0]:
        raise ValueError(
            "fitted and reference meshes must share topology: "
            f"{cur.shape[0]} vs {ref.shape[0]} vertices"
        )
    if faces.shape[0] != ref_faces.shape[0]:
        raise ValueError(
            "fitted and reference meshes must share topology: "
            f"{faces.shape[0]} vs {ref_faces.shape[0]} faces"
        )

    # tau from the fitted mesh bbox (self-relative), matching the fit's frame.
    bbox = cur.amax(dim=0) - cur.amin(dim=0)
    L = float(bbox.max().item())
    inflate = inflate_tau * (L / 2048.0)

    owner = torch.arange(cur.shape[0], device=device)
    t1 = time.time()
    hit_seg, hit_face, crossed = segmentMeshIntersections(
        ref, cur, cur, faces, owner, inflate=inflate
    )
    scan_s = time.time() - t1

    n_vertices = int(crossed.sum().item())
    n_pairs = int(hit_seg.numel())
    n_faces = int(torch.unique(hit_face).numel()) if hit_face.numel() else 0

    return {
        "mesh": mesh_path,
        "reference": reference_path,
        "L": L,
        "inflate": inflate,
        "n_vertices": int(cur.shape[0]),
        "n_faces": int(faces.shape[0]),
        "trajectory_crossing_vertices": n_vertices,
        "trajectory_crossing_pairs": n_pairs,
        "trajectory_crossing_faces": n_faces,
        "trajectory_self_intersection_free": bool(n_vertices == 0),
        "load_seconds": round(load_s, 2),
        "scan_seconds": round(scan_s, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inflate_tau",
        type=float,
        default=0.0,
        help="inflate segment/triangle AABBs by inflate_tau * L/2048 for a "
        "conservative near-touch test",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = evaluateTrajectoryFile(
        args.mesh,
        args.reference,
        device=args.device,
        inflate_tau=args.inflate_tau,
    )
    print(json.dumps(report, indent=2))
    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
