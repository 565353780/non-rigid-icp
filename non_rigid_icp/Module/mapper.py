import os
import numpy as np
import open3d as o3d
from typing import Union
from copy import deepcopy

from non_rigid_icp.Method.icp import icp
from non_rigid_icp.Method.nricp import nonrigidIcp
from non_rigid_icp.Method.render import renderMeshPair


class Mapper(object):
    def __init__(self) -> None:
        return

    def mapMesh(
        self,
        source_mesh_file_path: str,
        target_mesh_file_path: str,
        render: bool = False,
    ) -> Union[o3d.geometry.TriangleMesh, None]:
        if not os.path.exists(source_mesh_file_path):
            print('[ERROR][Mapper::mapMesh]')
            print('\t source mesh file not exist!')
            print('\t source_mesh_file_path:', source_mesh_file_path)
            return None

        if not os.path.exists(target_mesh_file_path):
            print('[ERROR][Mapper::mapMesh]')
            print('\t target mesh file not exist!')
            print('\t target_mesh_file_path:', target_mesh_file_path)
            return None

        sourcemesh = o3d.io.read_triangle_mesh(source_mesh_file_path)
        targetmesh = o3d.io.read_triangle_mesh(target_mesh_file_path)
        sourcemesh.compute_vertex_normals()
        targetmesh.compute_vertex_normals()

        if render:
            renderMeshPair(sourcemesh, targetmesh)

        initial_guess = np.eye(4)
        affine_transform = icp(sourcemesh,targetmesh,initial_guess)

        refined_sourcemesh = deepcopy(sourcemesh)
        refined_sourcemesh.transform(affine_transform)
        refined_sourcemesh.compute_vertex_normals()

        if render:
            renderMeshPair(refined_sourcemesh, targetmesh)

        deformed_mesh = nonrigidIcp(refined_sourcemesh,targetmesh)

        if render:
            renderMeshPair(deformed_mesh, targetmesh)

        return deformed_mesh
