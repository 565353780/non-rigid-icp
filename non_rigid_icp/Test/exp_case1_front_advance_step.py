"""Validate the front-advancing self-intersection-free single step on case1.

Runs ONE front-advancing projection step on the case1 source/target cropped to
the two ROI bboxes, and reports bbox Chamfer/F1, the residual before/after, the
number of advancing rounds + pull-backs, and -- the whole point -- the
authoritative new triangle-triangle self-intersection count. The earlier
"static-mesh cap" left ~10M new self-intersections; this step should drive that
to ~0 while keeping F1 at the same level.

Run (flux env, GPU 2):
  CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.exp_case1_front_advance_step
"""

from non_rigid_icp.Demo.watertight_fitter import data_dict
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.watertight_fitter import WatertightFitter


ROI_BBOXES = (
    {"name": "bbox_0", "center": (-0.02, 0.23, 0.01), "edge": 0.2},
    {"name": "bbox_1", "center": (-0.01, -0.12, 0.08), "edge": 0.2},
)


def main():
    src_path, tgt_path = data_dict["watertight_case1"]
    fitter = WatertightFitter(
        device="cuda",
        enable_self_collision_guard=False,
        enable_sheet_guard=False,
        enable_inversion_guard=False,
        enable_trajectory_guard=True,
        trajectory_seg_chunk=200000,
        max_subdivisions=0,
        eval_bboxes=ROI_BBOXES,
        prefit_crop_bboxes=ROI_BBOXES,
        prefit_crop_mode="centroid",
        save_result_folder_path="./output/case1_front_advance/",
    )
    fitter.loadMeshes(Mesh(src_path), Mesh(tgt_path))
    import os as _os
    backoff = float(_os.environ.get("RELAX_BACKOFF", "0.5"))
    relax_iters = int(_os.environ.get("RELAX_ITERS", "40"))
    rec = fitter.fitFrontAdvancingSingleStep(
        backoff=backoff,
        relax_iters=relax_iters,
        save_folder="./output/case1_front_advance/",
        crop_eval=True,
        compute_chamfer=False,
    )
    print("\n==== summary ====")
    for k in (
        "n_vertices", "relax_iters", "pen_pairs_start", "pen_pairs_end",
        "backed_off_vertices", "pinned_vertices", "mean_alpha",
        "fit_residual_before_tau", "fit_residual_after_tau",
        "bbox_0_crop_f1", "bbox_1_crop_f1",
        "tri_intersecting_pairs_new", "tri_intersecting_pairs_total",
        "seconds",
    ):
        print(f"  {k} = {rec.get(k)}")


if __name__ == "__main__":
    main()
