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

    if both_normal_exists:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    reg_p2p = o3d.pipelines.registration.registration_icp(
            sourceply, targetply, threshold, trans_init, estimation_method)

    return reg_p2p.transformation

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
