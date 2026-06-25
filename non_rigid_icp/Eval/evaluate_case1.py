import json
import argparse
import numpy as np

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.sampling import sampleMeshSurface, toTargetFrameTransform
from non_rigid_icp.Metric.chamfer import computeChamferMetrics, computeF1AtThreshold


def evaluate(
    source_path: str,
    target_path: str,
    samples: int = 2000000,
    device: str = "cuda",
    seed: int = 0,
    use_vertices: bool = False,
) -> dict:
    """Evaluate a fitted/source mesh against the target mesh.

    Metrics are computed in the ORIGINAL target coordinate frame:
      - Chamfer distance (bidirectional, L1 and L2)
      - F1-Score at tau = L/2048, where L is the largest bbox edge of the target.

    Note: F1 at this strict threshold is sensitive to sampling density. Use a
    high sample count (>= 2M) or --use_vertices for a faithful estimate.
    """
    source_mesh = Mesh(source_path)
    target_mesh = Mesh(target_path)

    _, _, L = toTargetFrameTransform(target_mesh.vertices)
    tau = L / 2048.0

    if use_vertices:
        src_pts = np.asarray(source_mesh.vertices, dtype=np.float32)
        tgt_pts = np.asarray(target_mesh.vertices, dtype=np.float32)
    else:
        src_pts = sampleMeshSurface(source_mesh, samples, seed=seed)
        tgt_pts = sampleMeshSurface(target_mesh, samples, seed=seed + 1)

    chamfer = computeChamferMetrics(src_pts, tgt_pts, device=device)
    f1 = computeF1AtThreshold(src_pts, tgt_pts, tau, device=device)

    metrics = {
        "source": source_path,
        "target": target_path,
        "L": L,
        "tau": tau,
        "n_source_points": int(src_pts.shape[0]),
        "n_target_points": int(tgt_pts.shape[0]),
        **chamfer,
        **f1,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--target",
        default="/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_gen.glb",
    )
    parser.add_argument("--samples", type=int, default=2000000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_vertices", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metrics = evaluate(
        args.source,
        args.target,
        samples=args.samples,
        device=args.device,
        seed=args.seed,
        use_vertices=args.use_vertices,
    )
    print(json.dumps(metrics, indent=2))
    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
