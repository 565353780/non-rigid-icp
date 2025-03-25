import torch
import numpy as np
import open3d as o3d
from typing import Union

from non_rigid_icp.Method.trans import toNumpy


def toPointCloud(points: Union[torch.Tensor, np.ndarray]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(toNumpy(points))
    return pcd
