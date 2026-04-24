import numpy as np
import open3d as o3d
from typing import Union
from copy import deepcopy


def pointsICP(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    source_normals: Union[np.ndarray, None]=None,
    target_normals: Union[np.ndarray, None]=None,
    threshold: float = 0.02,
    trans_init: np.ndarray=np.eye(4),
) -> np.ndarray:
    sourceply =  o3d.geometry.PointCloud()
    targetply =  o3d.geometry.PointCloud()
    sourceply.points = o3d.utility.Vector3dVector(source_pts)
    targetply.points = o3d.utility.Vector3dVector(target_pts)

    both_normal_exists = True
    if source_normals is not None:
        sourceply.normals = o3d.utility.Vector3dVector(source_normals)
    else:
        both_normal_exists = False

    if target_normals is not None:
        targetply.normals = o3d.utility.Vector3dVector(target_normals)
    else:
        both_normal_exists = False

    # 计算 source / target 的 AABB，按 Open3D 约定（齐次坐标左乘，p' = T @ p）构造
    # 各向同性缩放 + 平移矩阵，使 sourceply 的 bbox 与 targetply 的 bbox 最接近：
    #   p' = s * (p - c_s) + c_t = s * p + (c_t - s * c_s)
    # 其中 s 为两个 bbox 对角线长度之比。
    source_bbox = sourceply.get_axis_aligned_bounding_box()
    target_bbox = targetply.get_axis_aligned_bounding_box()
    source_center = np.asarray(source_bbox.get_center(), dtype=np.float64)
    target_center = np.asarray(target_bbox.get_center(), dtype=np.float64)
    source_diag = float(np.linalg.norm(np.asarray(source_bbox.get_extent(), dtype=np.float64)))
    target_diag = float(np.linalg.norm(np.asarray(target_bbox.get_extent(), dtype=np.float64)))
    if source_diag > 1e-12 and target_diag > 1e-12:
        scale = target_diag / source_diag
    else:
        scale = 1.0

    bbox_transform = np.eye(4, dtype=np.float64)
    bbox_transform[:3, :3] *= scale
    bbox_transform[:3, 3] = target_center - scale * source_center

    # 将该 bbox 对齐变换作用到 source 上，再执行 ICP
    sourceply.transform(bbox_transform)
    if both_normal_exists:
        # 各向同性缩放不改变法向方向，但 transform 会按缩放因子改变其模长，重新归一化
        sourceply.normalize_normals()

    if both_normal_exists:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    reg_p2p = o3d.pipelines.registration.registration_icp(
            sourceply, targetply, threshold, trans_init, estimation_method)

    # ICP 结果作用在已做 bbox 预对齐后的点上，需将两步变换复合：先 bbox，后 ICP
    icp_transform = np.asarray(reg_p2p.transformation, dtype=np.float64)
    final_transform = icp_transform @ bbox_transform

    return final_transform

def icp(
    source: Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud],
    target: Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud],
    threshold: float = 0.02,
    trans_init: np.ndarray=np.eye(4),
) -> Union[np.ndarray, None]:
    copied_source = deepcopy(source)
    copied_target = deepcopy(target)

    if isinstance(source, o3d.geometry.TriangleMesh):
        copied_source.compute_vertex_normals()
        source_pts = np.asarray(copied_source.vertices)
        source_normals = np.asarray(copied_source.vertex_normals)
    elif isinstance(source, o3d.geometry.PointCloud):
        source_pts = np.asarray(copied_source.points)
        source_normals = None
    else:
        print('[ERROR][icp::icp]')
        print('\t source data type not valid!')
        return None

    if isinstance(target, o3d.geometry.TriangleMesh):
        copied_target.compute_vertex_normals()
        target_pts = np.asarray(copied_target.vertices)
        target_normals = np.asarray(copied_target.vertex_normals)
    elif isinstance(target, o3d.geometry.PointCloud):
        target_pts = np.asarray(copied_target.points)
        target_normals = None
    else:
        print('[ERROR][icp::icp]')
        print('\t target data type not valid!')
        return None

    transformation = pointsICP(
        source_pts,
        target_pts,
        source_normals,
        target_normals,
        threshold,
        trans_init,
    )

    return transformation
