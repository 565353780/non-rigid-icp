import os
import open3d as o3d
from typing import Union
from copy import deepcopy

from non_rigid_icp.Method.icp import icp
from non_rigid_icp.Method.nricp import nonrigidIcp
from non_rigid_icp.Method.render import renderColoredMeshes


class Mapper(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def mapMesh(
        source_mesh_file_path: str,
        target_mesh_file_path: str,
        use_non_rigid_icp: bool=True,
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

        if render:
            renderColoredMeshes(
                [
                    sourcemesh,
                    targetmesh,
                ],
                [
                    [0.9, 0.1, 0.1],
                    [0.1, 0.9, 0.1],
                ]
            )

        affine_transform = icp(sourcemesh,targetmesh)

        refined_sourcemesh = deepcopy(sourcemesh)
        refined_sourcemesh.transform(affine_transform)
        refined_sourcemesh.compute_vertex_normals()

        if render:
            renderColoredMeshes(
                [
                    refined_sourcemesh,
                    targetmesh,
                ],
                [
                    [0.1, 0.1, 0.9],
                    [0.1, 0.9, 0.1],
                ]
            )

        if not use_non_rigid_icp:
            return refined_sourcemesh

        deformed_mesh = nonrigidIcp(refined_sourcemesh,targetmesh)

        if render:
            renderColoredMeshes(
                [
                    sourcemesh,
                    deformed_mesh,
                    targetmesh,
                ],
                [
                    [0.9, 0.1, 0.1],
                    [0.1, 0.1, 0.9],
                    [0.1, 0.9, 0.1],
                ]
            )

        return deformed_mesh
