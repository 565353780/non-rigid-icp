import os
import cv2
import torch
import numpy as np
import open3d as o3d
from tqdm import trange
from typing import Union

if torch.cuda.is_available():
    from non_rigid_icp.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
else:
    from non_rigid_icp.Lib.chamfer3D.chamfer_python import distChamfer

from non_rigid_icp.Constraint.target_points import TargetPointsConstraint
from non_rigid_icp.Constraint.target_vertices import TargetVerticesConstraint
from non_rigid_icp.Constraint.fixed_vertices import FixedVerticesConstraint
from non_rigid_icp.Constraint.vertex_group import VertexGroupConstraint
from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Loss.masked_dist import toMaskedDistLoss
from non_rigid_icp.Loss.laplacian import toLaplacian, toLaplacianLoss
from non_rigid_icp.Method.icp import icp
from non_rigid_icp.Method.trans import toNumpy, toPointCloud
from non_rigid_icp.Method.time import getCurrentTime
from non_rigid_icp.Method.render import renderGeometryImages, renderConstraints
from non_rigid_icp.Method.trans import (
    toMesh,
    toTensor,
)
from non_rigid_icp.Method.video import toVideo
from non_rigid_icp.Model.deform import DeformModel
from non_rigid_icp.Metric.chamfer import toL1ChamferDistance
from non_rigid_icp.Module.logger import Logger


class OptimalMapper(object):
    def __init__(
        self,
        inner_iter: int = 50,
        outer_iter: int = 200,
        milestones: list = [50, 80, 100, 110, 120, 130, 140, 150],
        masked_dist_thresh: float = 0.04,
        masked_dist_weight: float = 1.0,
        stiffness_weights: list = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2, 0],
        laplacian_weight: float = 1.0,
        target_vertices_weight: float = 1.0,
        device: str = "cpu",
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        render: bool = False,
    ) -> None:
        self.width = 1920 // 3
        self.height = 1080 // 2

        self.inner_iter = inner_iter
        self.outer_iter = outer_iter
        self.milestones = milestones

        self.masked_dist_thresh = masked_dist_thresh
        self.masked_dist_weight = masked_dist_weight
        self.stiffness_weights = stiffness_weights
        self.laplacian_weight = laplacian_weight
        self.target_vertices_weight = target_vertices_weight

        self.device = device

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.render = render

        self.logger = Logger()

        if torch.cuda.is_available() and device != "cpu":
            self.chamfer_func = chamfer_3DDist()
        else:
            self.chamfer_func = distChamfer

        self.initRecords()

        self.template_mesh = Mesh()

        # constraints
        self.target_points_constraint = TargetPointsConstraint()
        self.target_vertices_constraint = TargetVerticesConstraint()
        self.vertex_group_constraint = VertexGroupConstraint()
        self.fixed_vertex_constraint = FixedVerticesConstraint()

        # tmp
        self.save_deform_image_idx = 0
        return

    def isValid(self) -> bool:
        if not self.template_mesh.isValid():
            print("[ERROR][OptimalMapper::isValid]")
            print(
                "\t isValid failed for template mesh! please load template mesh first!"
            )
            return False

        return True

    def initRecords(self) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def addTargetPointsConstraint(self, target_points: np.ndarray) -> bool:
        return self.target_points_constraint.addConstraint(target_points)

    def addTargetVerticesConstraint(
        self, vertex_idxs: np.ndarray, target_vertices: np.ndarray
    ) -> bool:
        return self.target_vertices_constraint.addConstraint(vertex_idxs, target_vertices)

    def addVertexGroupConstraint(self, group_id: int, vertex_idxs: np.ndarray) -> bool:
        return self.vertex_group_constraint.addConstraint(group_id, vertex_idxs)

    def addFixedVertexConstraint(
        self, vertex_idxs: np.ndarray, target_positions: np.ndarray
    ) -> bool:
        return self.fixed_vertex_constraint.addConstraint(vertex_idxs, target_positions)

    def loadTemplateMesh(self, vertices: np.ndarray, triangles: np.ndarray) -> bool:
        self.template_mesh.vertices = vertices
        self.template_mesh.triangles = triangles

        if self.render:
            self.recordGeometry("Template", vertices, triangles)
        return True

    def updateTemplateVertices(
        self, new_vertices: Union[torch.Tensor, np.ndarray]
    ) -> bool:
        self.template_mesh.vertices = toNumpy(new_vertices, np.float32)
        return True

    def estimateInitPose(
        self,
    ) -> o3d.geometry.TriangleMesh:
        if not self.isValid():
            print("[ERROR][OptimalMapper::estimateInitPose]")
            print("\t isValid failed!")
            return False

        if not self.target_points_constraint.isValid():
            print("[ERROR][OptimalMapper::estimateInitPose]")
            print("\t target_points_constraint is not valid!")
            return False

        template_mesh = self.template_mesh.toO3DMesh()
        target_pcd = toPointCloud(self.target_points_constraint.getConstraint())

        transformation = icp(template_mesh, target_pcd)
        assert transformation is not None

        template_mesh.transform(transformation)

        self.updateTemplateVertices(template_mesh.vertices)

        if self.render:
            self.recordGeometry(
                "ICP", self.template_mesh.vertices, self.template_mesh.triangles
            )
        return True

    def map(self) -> bool:
        if not self.isValid():
            print("[ERROR][OptimalMapper::map]")
            print("\t isValid failed!")
            return False

        if self.target_points_constraint.isValid():
            self.target_points_constraint.updateTensor(self.device)

            if self.render:
                self.recordGeometry(
                    "Target", self.target_points_constraint.getConstraint()
                )

            if not self.estimateInitPose():
                print("[ERROR][OptimalMapper::map]")
                print("\t estimateInitPose failed!")
                return False

        if self.target_vertices_constraint.isValid():
            self.target_vertices_constraint.updateTensor(self.device)

        save_deformed_image_folder_path = self.save_result_folder_path + "DeformedMesh/"
        os.makedirs(save_deformed_image_folder_path, exist_ok=True)

        loop = trange(self.outer_iter)
        w_idx = 0

        template_triangles = toTensor(
            self.template_mesh.triangles, self.device, torch.int64
        )

        source_laplacian = toLaplacian(
            toTensor(self.template_mesh.vertices, self.device, torch.float32),
            template_triangles,
        )

        vertex_group_idxs = None
        if self.vertex_group_constraint.isValid():
            vertex_group_idxs = self.vertex_group_constraint.getConstraint(
                self.template_mesh.vertices.shape[0]
            )

        fixed_vertex_idxs = None
        fixed_target_positions = None
        if self.fixed_vertex_constraint.isValid():
            fixed_vertex_idxs, fixed_target_positions = (
                self.fixed_vertex_constraint.getConstraint()
            )

        """
        renderConstraints(
            self.template_mesh.toO3DPcd(),
            toPointCloud(self.target_points_constraint.getConstraint()),
            fixed_vertex_idxs,
            fixed_target_positions,
            vertex_group_idxs,
        )
        """

        deform_model = DeformModel(
            self.template_mesh,
            self.device,
            fixed_vertex_idxs=fixed_vertex_idxs,
            fixed_target_positions=fixed_target_positions,
            vertex_group_idxs=vertex_group_idxs,
        )
        deform_model.setDeformGradState(True)

        deform_model.moveToLandMark()

        optimizer = torch.optim.AdamW(
            [
                deform_model.deform_field.rotate_matrixs,
                deform_model.deform_field.translates,
            ],
            lr=1e-4,
            amsgrad=True,
        )

        is_target_points_constraint_valid = self.target_points_constraint.isValid()
        is_target_vertices_constraint_valid = self.target_vertices_constraint.isValid()

        step = 0
        for i in loop:
            if i == 0:
                inner_loop = range(4)
            else:
                inner_loop = range(self.inner_iter)

            new_deformed_verts = deform_model.deform()

            if is_target_points_constraint_valid:
                idx1 = self.chamfer_func(
                    new_deformed_verts.unsqueeze(0),
                    self.target_points_constraint.points_tensor,
                )[2]
                close_points = self.target_points_constraint.points_tensor[
                    0, idx1.squeeze(0)
                ]

            for _ in inner_loop:
                optimizer.zero_grad()

                new_deformed_verts = deform_model.deform()

                masked_dist_loss = (
                    torch.tensor(0.0)
                    .type(new_deformed_verts.dtype)
                    .to(new_deformed_verts.device)
                )
                if is_target_points_constraint_valid:
                    if self.masked_dist_weight > 0:
                        masked_dist_loss = toMaskedDistLoss(
                            new_deformed_verts, close_points, self.masked_dist_thresh
                        )

                rotate_stiffness_loss = (
                    torch.tensor(0.0)
                    .type(new_deformed_verts.dtype)
                    .to(new_deformed_verts.device)
                )
                translate_stiffness_loss = (
                    torch.tensor(0.0)
                    .type(new_deformed_verts.dtype)
                    .to(new_deformed_verts.device)
                )
                stiffness_weight = self.stiffness_weights[w_idx]
                if stiffness_weight > 0:
                    rotate_stiffness, translate_stiffness = deform_model.stiffness()
                    rotate_stiffness_loss = (
                        self.stiffness_weights[w_idx] * rotate_stiffness
                    )
                    translate_stiffness_loss = (
                        self.stiffness_weights[w_idx] * translate_stiffness
                    )

                laplacian_loss = (
                    torch.tensor(0.0)
                    .type(new_deformed_verts.dtype)
                    .to(new_deformed_verts.device)
                )
                if self.laplacian_weight > 0:
                    laplacian_loss = toLaplacianLoss(
                        new_deformed_verts, template_triangles, source_laplacian
                    )

                    laplacian_loss = self.laplacian_weight * laplacian_loss

                target_vertices_loss = (
                    torch.tensor(0.0)
                    .type(new_deformed_verts.dtype)
                    .to(new_deformed_verts.device)
                )
                if is_target_vertices_constraint_valid:
                    if self.target_vertices_weight > 0:
                        target_vertices_loss = (
                            self.target_vertices_constraint.computeL1Loss(
                                new_deformed_verts
                            )
                        )
                        target_vertices_loss = (
                            self.target_vertices_weight * target_vertices_loss
                        )

                # loss = torch.sqrt(masked_dist_loss + stiffness_loss) + laplacian_loss
                loss = (
                    masked_dist_loss
                    + rotate_stiffness_loss
                    + translate_stiffness_loss
                    + laplacian_loss
                    + target_vertices_loss
                )
                loss.backward()
                optimizer.step()

                deform_model.moveToLandMark()

                self.logger.addScalar(
                    "Loss/RotateStiffness", rotate_stiffness_loss.item(), step
                )
                self.logger.addScalar(
                    "Loss/TranslateStiffness", translate_stiffness_loss.item(), step
                )
                self.logger.addScalar(
                    "Loss/MatchingDist", masked_dist_loss.item(), step
                )
                self.logger.addScalar("Loss/Laplacian", laplacian_loss.item(), step)
                self.logger.addScalar(
                    "Loss/TargetVertices", target_vertices_loss.item(), step
                )

                step += 1

            if is_target_points_constraint_valid:
                dist1, dist2 = self.chamfer_func(
                    new_deformed_verts.unsqueeze(0),
                    self.target_points_constraint.points_tensor,
                )[:2]
                l1_chamfer = toL1ChamferDistance(dist1, dist2)
                self.logger.addScalar("Metric/L1-Chamfer", l1_chamfer, step)

            if self.render:
                new_deformed_verts = deform_model.deform()
                new_deformed_mesh = toMesh(new_deformed_verts, template_triangles)
                new_deformed_mesh.compute_vertex_normals()
                image = renderGeometryImages(
                    new_deformed_mesh, width=self.width, height=self.height
                )
                self.logger.addImage("DeformedMesh", image, step)
                if self.save_result_folder_path is not None:
                    cv2.imwrite(
                        save_deformed_image_folder_path
                        + str(self.save_deform_image_idx)
                        + ".jpg",
                        image,
                    )
                    self.save_deform_image_idx += 1

            if i in self.milestones:
                w_idx += 1

        new_deformed_verts = deform_model.deform()
        self.updateTemplateVertices(new_deformed_verts)

        if self.render:
            save_video_file_path = self.save_result_folder_path + "Deform.mp4"
            fps = 10
            toVideo(save_deformed_image_folder_path, save_video_file_path, fps)

            self.logger.addVideoFile("Deform", save_video_file_path, fps)
        return True

    def toDeformedTemplateMesh(self) -> o3d.geometry.TriangleMesh:
        return self.template_mesh.clone()

    def recordGeometry(
        self, name: str, vertices: np.ndarray, triangles: Union[np.ndarray, None] = None
    ):
        if triangles is None:
            geometry = toPointCloud(vertices)
            geometry.estimate_normals()
        else:
            geometry = toMesh(vertices, triangles)
            geometry.compute_vertex_normals()

        if self.render:
            image = renderGeometryImages(geometry, width=self.width, height=self.height)
            self.logger.addImage(name, image)
            if self.save_result_folder_path is not None:
                cv2.imwrite(self.save_result_folder_path + name + ".jpg", image)
        return True
