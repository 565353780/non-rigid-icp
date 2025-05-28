import os
import trimesh
import numpy as np
import open3d as o3d
from copy import deepcopy
from typing import Union

from non_rigid_icp.Method.io import loadMeshFile, loadPLYAttributes, loadConstrains
from non_rigid_icp.Method.path import createFileFolder, removeFile, renameFile
from non_rigid_icp.Method.render import renderGeometries


class Mesh(object):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
    ) -> None:
        self.triangle_normals = None
        self.triangles = None
        self.vertex_colors = None
        self.vertex_normals = None
        self.vertices = None

        self.attributes = None
        self.constrains = {}

        if mesh_file_path is not None:
            self.loadMesh(mesh_file_path)
        return

    def reset(self):
        self.triangle_normals = None
        self.triangles = None
        self.vertex_colors = None
        self.vertex_normals = None
        self.vertices = None

        self.attributes = None
        return True

    @classmethod
    def from_o3d(cls, o3d_mesh: o3d.geometry.TriangleMesh):
        mesh = cls()
        mesh.loadO3DMeshProperties(o3d_mesh)
        return mesh

    def loadO3DMeshProperties(self, o3d_mesh: o3d.geometry.TriangleMesh) -> bool:
        self.triangle_normals = np.asarray(o3d_mesh.triangle_normals)
        self.triangles = np.asarray(o3d_mesh.triangles)
        self.vertex_colors = np.asarray(o3d_mesh.vertex_colors)
        self.vertex_normals = np.asarray(o3d_mesh.vertex_normals)
        self.vertices = np.asarray(o3d_mesh.vertices)
        return True

    def loadMesh(self, mesh_file_path: str):
        self.reset()

        o3d_mesh = loadMeshFile(mesh_file_path)

        if o3d_mesh is None:
            print("[ERROR][Mesh::loadMesh]")
            print("\t loadMeshFile failed!")
            return False

        if not self.loadO3DMeshProperties(o3d_mesh):
            print("[ERROR][Mesh::loadMesh]")
            print("\t loadO3DMeshProperties failed!")
            return False

        if mesh_file_path.endswith(".ply"):
            self.attributes = loadPLYAttributes(mesh_file_path)
            self.constrains = loadConstrains(self.attributes)

        return True

    def isValid(self, output_info=False):
        if self.vertices is None:
            if output_info:
                print("[ERROR][Mesh::isValid]")
                print("\t vertices is None! please load mesh first!")
            return False

        if self.triangles is None:
            if output_info:
                print("[ERROR][Mesh::isValid]")
                print("\t triangles is None! please load mesh first!")
            return False

        if self.vertices.shape[0] == 0:
            if output_info:
                print("[ERROR][Mesh::isValid]")
                print("\t vertices is empty! please check this mesh!")
            return False

        return True

    def center(self) -> np.ndarray:
        min_bound = np.min(self.vertices, axis=0)
        max_bound = np.max(self.vertices, axis=0)

        center = (min_bound + max_bound) / 2.0
        return center

    def length(self) -> float:
        min_bound = np.min(self.vertices, axis=0)
        max_bound = np.max(self.vertices, axis=0)
        length = np.max(max_bound - min_bound)
        return length

    def normalize(self) -> bool:
        scale = 0.9 / self.length()
        self.vertices = (self.vertices - self.center()) * scale
        return True

    def transform(
        self, center: np.ndarray, scale: float, is_inverse: bool = False
    ) -> bool:
        if is_inverse:
            self.vertices = self.vertices / scale + center
        else:
            self.vertices = (self.vertices - center) * scale

        return True

    def toO3DPcd(self) -> o3d.geometry.PointCloud:
        o3d_pcd = o3d.geometry.PointCloud()
        if self.vertex_colors is not None:
            o3d_pcd.colors = o3d.utility.Vector3dVector(self.vertex_colors)
        if self.vertex_normals is not None:
            o3d_pcd.normals = o3d.utility.Vector3dVector(self.vertex_normals)
        if self.vertices is not None:
            o3d_pcd.points = o3d.utility.Vector3dVector(self.vertices)
        return o3d_pcd

    def toO3DMesh(self) -> o3d.geometry.TriangleMesh:
        o3d_mesh = o3d.geometry.TriangleMesh()
        if self.triangle_normals is not None:
            o3d_mesh.triangle_normals = o3d.utility.Vector3dVector(
                self.triangle_normals
            )
        if self.triangles is not None:
            o3d_mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        if self.vertex_colors is not None:
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        if self.vertex_normals is not None:
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        if self.vertices is not None:
            o3d_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        return o3d_mesh

    def toO3DTensorMesh(self) -> o3d.t.geometry.TriangleMesh:
        return o3d.t.geometry.TriangleMesh.from_legacy(self.toO3DMesh())

    def toTrimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=deepcopy(self.vertices),
            faces=deepcopy(self.triangles),
            face_normals=deepcopy(self.triangle_normals),
            vertex_normals=deepcopy(self.vertex_normals),
        )

    def toO3DABB(self) -> o3d.geometry.AxisAlignedBoundingBox:
        return self.toO3DMesh().get_axis_aligned_bounding_box()

    def toABBMaxBound(self) -> np.ndarray:
        return np.array(self.toO3DABB().get_max_bound(), dtype=np.float64)

    def toABBLength(self) -> float:
        return np.linalg.norm(self.toABBMaxBound(), ord=2)

    def toBoundaryIdxs(self) -> np.ndarray:
        half_edge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(
            self.toO3DMesh()
        )
        return np.array(half_edge_mesh.get_boundaries(), dtype=int).reshape(-1)

    def save(self, save_mesh_file_path: str, overwrite: bool = False) -> bool:
        if not overwrite:
            if os.path.exists(save_mesh_file_path):
                return True

        removeFile(save_mesh_file_path)

        o3d_mesh = self.toO3DMesh()

        tmp_save_mesh_file_path = (
            save_mesh_file_path[:-4] + "_tmp" + save_mesh_file_path[-4:]
        )

        createFileFolder(save_mesh_file_path)

        o3d.io.write_triangle_mesh(tmp_save_mesh_file_path, o3d_mesh, write_ascii=True)

        renameFile(tmp_save_mesh_file_path, save_mesh_file_path)
        return True

    def render(self):
        if not self.isValid(True):
            print("[ERROR][Mesh::render]")
            print("\t isValid failed!")
            return False

        renderGeometries(self.toO3DMesh(), "Mesh")
        return True
