import numpy as np
import open3d as o3d
from typing import Union
from copy import deepcopy

def toPaintedMesh(
    mesh: o3d.geometry.TriangleMesh,
    color: Union[np.ndarray, list] = [0.1, 0.1, 0.9],
) -> o3d.geometry.TriangleMesh:
    painted_mesh = deepcopy(mesh)
    painted_mesh.paint_uniform_color(color)
    return painted_mesh

def renderColoredMeshes(
    mesh_list: list,
    color_list: list,
) -> bool:
    painted_mesh_list = []

    for mesh, color in zip(mesh_list, color_list):
        painted_mesh = toPaintedMesh(mesh, color)

        painted_mesh_list.append(painted_mesh)

    o3d.visualization.draw_geometries(painted_mesh_list)
    return True
