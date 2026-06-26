"""Authoritative self-intersection evaluation for any mesh file.

Loads a mesh, runs the global grid-hash broad phase + exact narrow phase, and
reports the number of intersecting non-adjacent triangle pairs, the number of
faces involved, and a connected-component clustering of the intersecting region.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Eval.self_intersection_eval \
        --mesh output/case1_refine/fitted_mesh.ply --out si_report.json
"""

import json
import time
import argparse
import numpy as np
import torch

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.self_intersection import (
    selfIntersectionReport,
    findSelfIntersections,
)


def evaluateMeshFile(
    mesh_path: str,
    device: str = "cuda",
    inflate_tau: float = 0.0,
    exclude_ring: int = 1,
    dump_pairs_path: str = None,
) -> dict:
    t0 = time.time()
    mesh = Mesh(mesh_path)
    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.asarray(mesh.triangles, dtype=np.int64)).to(device)
    load_s = time.time() - t0

    # tau from this mesh's own bbox (self-relative); callers may override inflate
    bbox = vertices.amax(dim=0) - vertices.amin(dim=0)
    L = float(bbox.max().item())
    inflate = inflate_tau * (L / 2048.0)

    t1 = time.time()
    report = selfIntersectionReport(
        vertices, faces, inflate=inflate, exclude_ring=exclude_ring, cluster=True
    )
    scan_s = time.time() - t1

    report.update(
        {
            "mesh": mesh_path,
            "L": L,
            "inflate": inflate,
            "exclude_ring": exclude_ring,
            "load_seconds": round(load_s, 2),
            "scan_seconds": round(scan_s, 2),
        }
    )

    if dump_pairs_path is not None and report["num_intersecting_pairs"] > 0:
        inter = findSelfIntersections(
            vertices, faces, inflate=inflate, exclude_ring=exclude_ring
        )
        np.save(dump_pairs_path, inter.detach().cpu().numpy())
        report["dumped_pairs"] = dump_pairs_path

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inflate_tau",
        type=float,
        default=0.0,
        help="inflate AABBs by inflate_tau * L/2048 to also surface near-touch pairs",
    )
    parser.add_argument("--exclude_ring", type=int, default=1)
    parser.add_argument("--dump_pairs", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = evaluateMeshFile(
        args.mesh,
        device=args.device,
        inflate_tau=args.inflate_tau,
        exclude_ring=args.exclude_ring,
        dump_pairs_path=args.dump_pairs,
    )
    print(json.dumps(report, indent=2))
    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
