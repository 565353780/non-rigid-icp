import numpy as np
import open3d as o3d
from copy import deepcopy


def icp(source,target,trans_init=np.eye(4)):
    sourcemesh=deepcopy(source)
    targetmesh=deepcopy(target)
    sourceply =  o3d.geometry.PointCloud()
    targetply =  o3d.geometry.PointCloud()
    sourcemesh.compute_vertex_normals()
    targetmesh.compute_vertex_normals()
    sourceply.points = sourcemesh.vertices
    targetply.points = targetmesh.vertices
    sourceply.normals = sourcemesh.vertex_normals
    targetply.normals = targetmesh.vertex_normals

    threshold = 0.02
    reg_p2p = o3d.pipelines.registration.registration_icp(
            sourceply, targetply, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return reg_p2p.transformation
