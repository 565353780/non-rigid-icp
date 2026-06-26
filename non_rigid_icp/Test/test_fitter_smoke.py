"""End-to-end smoke test of the guarded + adaptively-subdivided fitter.

Uses small synthetic watertight meshes so it runs in seconds, while exercising
the full pipeline: rigid init -> optimize-to-plateau -> error localization ->
conforming subdivision -> self-collision guard -> metric-guarded evaluation.

    python -m non_rigid_icp.Test.test_fitter_smoke
"""

import numpy as np
import open3d as o3d

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


def _sphere(resolution: int, radius: float = 1.0, scale=(1.0, 1.0, 1.0), bump: float = 0.0):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    v = np.asarray(m.vertices)
    v = v * np.asarray(scale)[None, :]
    if bump != 0.0:
        # add a smooth localized bump so the target differs non-trivially
        r = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        mask = (v[:, 2:3] > 0.5 * radius).astype(np.float64)
        v = v + bump * mask * (v / r)
    m.vertices = o3d.utility.Vector3dVector(v)
    m.compute_vertex_normals()
    return m


def main():
    source_o3d = _sphere(resolution=20, radius=1.0)
    target_o3d = _sphere(resolution=24, radius=1.0, scale=(1.25, 1.0, 0.8), bump=0.15)

    source_mesh = Mesh.from_o3d(source_o3d)
    target_mesh = Mesh.from_o3d(target_o3d)

    # the sphere->ellipsoid deformation is many tau, so use a coarse outlier mask
    # schedule (the default tau-relative schedule is tuned for sources already
    # within ~16 tau of the target, like the watertight case).
    fitter = WatertightFitter(
        device="cuda",
        outer_iter=20,
        inner_iter=6,
        lr=1e-2,
        train_source_samples=20000,
        train_target_samples=120000,
        eval_samples=120000,
        laplacian_weight=20.0,
        mask_dist_schedule=[0.5, 0.3, 0.2, 0.1, 0.05],
        enable_self_collision_guard=True,
        collision_weight=50.0,
        collision_refresh_every=3,
        collision_check_every=3,
        max_subdivisions=2,
        refine_iter=10,
        plateau_window=3,
        plateau_rel_tol=1e-2,
        plateau_patience=2,
        error_quantile=0.85,
        save_result_folder_path=None,
        seed=0,
    )
    fitter.loadMeshes(source_mesh, target_mesh)
    result = fitter.fitAndEvaluate()

    print("baseline:", {k: round(v, 6) if isinstance(v, float) else v for k, v in result["baseline"].items()})
    print("fitted:  ", {k: round(v, 6) if isinstance(v, float) else v for k, v in result["fitted"].items()})
    print("kept:", result["kept"])
    print("refine_log:")
    for r in result["refine_log"]:
        print("   ", r)
    print("final new self-intersections:", result["final_new_self_intersections"])

    # assertions
    assert result["final_new_self_intersections"] == 0, "left a NEW self-intersection!"
    assert result["fitted"]["chamfer_l1"] <= result["baseline"]["chamfer_l1"] * 1.001, (
        "fit did not improve chamfer over rigid baseline"
    )
    # at least one subdivision round should have happened (plateau -> refine)
    assert len(result["refine_log"]) >= 1, "no adaptive subdivision triggered"
    # subdivision must increase resolution
    last = result["refine_log"][-1]
    assert last["faces_after"] > last["faces_before"], "subdivision did not add faces"
    print("\nFITTER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
