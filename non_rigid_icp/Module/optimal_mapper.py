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

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Loss.masked_dist import toMaskedDistLoss
from non_rigid_icp.Loss.laplacian import toLaplacian, toLaplacianLoss
from non_rigid_icp.Method.icp import icp
from non_rigid_icp.Method.trans import toNumpy
from non_rigid_icp.Method.path import createFileFolder, removeFile
from non_rigid_icp.Method.time import getCurrentTime
from non_rigid_icp.Method.render import renderGeometryImages
from non_rigid_icp.Method.trans import toMesh, toNormalizeTransform, toTensor, transGeometry, transPoints
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
        device: str='cpu',
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        render: bool=False,
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

        self.device = device

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.render = render

        self.logger = Logger()

        if torch.cuda.is_available() and device != 'cpu':
            self.chamfer_func = chamfer_3DDist()
        else:
            self.chamfer_func = distChamfer

        self.initRecords()

        self.gt_points = None
        self.gt_geometry = None
        self.gt_center = None
        self.gt_scale = None

        self.template_mesh = Mesh()

        # tmp
        self.save_deform_image_idx = 0
        return

    def isGTValid(self) -> bool:
        if self.gt_geometry is None:
            print('[ERROR][OptimalMapper::isGTValid]')
            print('\t gt geometry not exist! please load gt geometry first!')
            return False

        return True

    def isTemplateValid(self) -> bool:
        if not self.template_mesh.isValid():
            print('[ERROR][OptimalMapper::isTemplateValid]')
            print('\t isValid failed for template mesh! please load template mesh first!')
            return False

        return True

    def isValid(self) -> bool:
        return self.isGTValid() and self.isTemplateValid()

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

    def normalizeGTGeometry(self) -> bool:
        if not self.isGTValid():
            print('[ERROR][OptimalMapper::normalizeGTGeometry]')
            print('\t isGTValid failed!')
            return False

        if isinstance(self.gt_geometry, o3d.geometry.PointCloud):
            gt_points = np.asarray(self.gt_geometry.points)
        else:
            gt_points = np.asarray(self.gt_geometry.vertices)

        self.gt_center, self.gt_scale = toNormalizeTransform(gt_points)

        transGeometry(self.gt_geometry, self.gt_center, self.gt_scale)

        if isinstance(self.gt_geometry, o3d.geometry.TriangleMesh):
            self.gt_geometry.compute_vertex_normals()

        if isinstance(self.gt_geometry, o3d.geometry.PointCloud):
            self.gt_points = np.asarray(self.gt_geometry.points)
        else:
            self.gt_points = np.asarray(self.gt_geometry.vertices)

        self.gt_points = toTensor(self.gt_points, self.device).unsqueeze(0)

        if self.render:
            image = renderGeometryImages(
                self.gt_geometry,
                width=self.width,
                height=self.height)
            self.logger.addImage('GT', image)
            if self.save_result_folder_path is not None:
                cv2.imwrite(self.save_result_folder_path + 'GT.jpg', image)
        return True

    def normalizeTemplateMesh(self) -> bool:
        if not self.isTemplateValid():
            print('[ERROR][OptimalMapper::normalizeTemplateGeometry]')
            print('\t isTemplateValid failed!')
            return False

        self.template_mesh.normalize()

        if self.render:
            image = renderGeometryImages(
                self.template_mesh,
                width=self.width,
                height=self.height)
            self.logger.addImage('Template', image)
            if self.save_result_folder_path is not None:
                cv2.imwrite(self.save_result_folder_path + 'Template.jpg', image)
        return True

    def loadGTPcd(self, gt_pcd: o3d.geometry.PointCloud) -> bool:
        self.gt_geometry = gt_pcd
        return self.normalizeGTGeometry()

    def loadGTPcdFile(self, gt_pcd_file_path: str) -> bool:
        if not os.path.exists(gt_pcd_file_path):
            print('[ERROR][OptimalMapper::loadGTPcdFile]')
            print('\t gt pcd file not exist!')
            print('\t gt_pcd_file_path:', gt_pcd_file_path)
            return False

        gt_pcd = o3d.io.read_point_cloud(gt_pcd_file_path)
        return self.loadGTPcd(gt_pcd)

    def loadGTMeshFile(self, gt_mesh_file_path: str) -> bool:
        if not os.path.exists(gt_mesh_file_path):
            print('[ERROR][OptimalMapper::loadGTMeshFile]')
            print('\t gt mesh file not exist!')
            print('\t gt_mesh_file_path:', gt_mesh_file_path)
            return False

        self.gt_geometry = Mesh(gt_mesh_file_path)
        return self.normalizeGTGeometry()

    def loadTemplateMeshFile(self, template_mesh_file_path: str) -> bool:
        if not os.path.exists(template_mesh_file_path):
            print('[ERROR][OptimalMapper::loadTemplateMeshFile]')
            print('\t template mesh file not exist!')
            print('\t template_mesh_file_path:', template_mesh_file_path)
            return False

        self.template_mesh.loadMesh(template_mesh_file_path)
        self.normalizeTemplateMesh()
        return True

    def updateTemplateVertices(self, new_vertices: Union[torch.Tensor, np.ndarray]) -> bool:
        self.template_mesh.vertices = toNumpy(new_vertices, np.float32)
        return True

    def estimateInitPose(
        self,
    ) -> o3d.geometry.TriangleMesh:
        if not self.isValid():
            print('[ERROR][OptimalMapper::estimateInitPose]')
            print('\t isValid failed!')
            return False

        template_mesh = self.template_mesh.toO3DMesh()
        if isinstance(self.gt_geometry, o3d.geometry.PointCloud):
            gt_geometry = self.gt_geometry
        else:
            gt_geometry = self.gt_geometry.toO3DPcd()
        transformation = icp(template_mesh, gt_geometry)
        assert transformation is not None

        template_mesh.transform(transformation)

        self.updateTemplateVertices(template_mesh.vertices)

        if self.render:
            image = renderGeometryImages(
                self.template_mesh,
                width=self.width,
                height=self.height)
            self.logger.addImage('ICP', image)
            if self.save_result_folder_path is not None:
                cv2.imwrite(self.save_result_folder_path + 'ICP.jpg', image)
        return True

    def refineGeometry(
        self,
    ) -> bool:
        if not self.isValid():
            print('[ERROR][OptimalMapper::refineGeometry]')
            print('\t isValid failed!')
            return False

        save_deformed_image_folder_path = self.save_result_folder_path + 'DeformedMesh/'
        os.makedirs(save_deformed_image_folder_path, exist_ok=True)

        loop = trange(self.outer_iter)
        w_idx = 0

        template_triangles = toTensor(self.template_mesh.triangles, self.device, torch.int64)

        source_laplacian = toLaplacian(
            toTensor(self.template_mesh.vertices, self.device, torch.float32),
            template_triangles
        )

        deform_model = DeformModel(
            self.template_mesh,
            self.device,
            fixed_vertex_idxs=self.template_mesh.constrains['fixed_vertex_idxs'],
            fixed_target_positions=self.gt_geometry.vertices[self.template_mesh.constrains['fixed_vertex_idxs']],
            vertex_group_idxs=self.template_mesh.constrains['vertex_group_idxs'],
        )
        deform_model.setDeformGradState(True)

        deform_model.moveToLandMark()

        optimizer = torch.optim.AdamW([
            deform_model.deform_field.rotate_matrixs,
            deform_model.deform_field.translates,
        ], lr=1e-4, amsgrad=True)

        step = 0
        for i in loop:
            new_deformed_verts = deform_model.deform()

            idx1 = self.chamfer_func(new_deformed_verts.unsqueeze(0), self.gt_points)[2]

            if i == 0:
                inner_loop = range(4)
            else:
                inner_loop = range(self.inner_iter)

            close_points = self.gt_points[0, idx1.squeeze(0)]

            for _ in inner_loop:
                optimizer.zero_grad()

                new_deformed_verts = deform_model.deform()

                masked_dist_loss = torch.tensor(0.0).type(new_deformed_verts.dtype).to(new_deformed_verts.device)
                if self.masked_dist_weight > 0:
                    masked_dist_loss = toMaskedDistLoss(new_deformed_verts, close_points, self.masked_dist_thresh)

                rotate_stiffness_loss = torch.tensor(0.0).type(new_deformed_verts.dtype).to(new_deformed_verts.device)
                translate_stiffness_loss = torch.tensor(0.0).type(new_deformed_verts.dtype).to(new_deformed_verts.device)
                stiffness_weight = self.stiffness_weights[w_idx]
                if stiffness_weight > 0:
                    rotate_stiffness, translate_stiffness = deform_model.stiffness()
                    rotate_stiffness_loss = self.stiffness_weights[w_idx] * rotate_stiffness
                    translate_stiffness_loss = self.stiffness_weights[w_idx] * translate_stiffness

                laplacian_loss = torch.tensor(0.0).type(new_deformed_verts.dtype).to(new_deformed_verts.device)
                if self.laplacian_weight > 0:
                    laplacian_loss = toLaplacianLoss(
                        new_deformed_verts, template_triangles, source_laplacian)

                    laplacian_loss = self.laplacian_weight * laplacian_loss

                # loss = torch.sqrt(masked_dist_loss + stiffness_loss) + laplacian_loss
                loss = masked_dist_loss + rotate_stiffness_loss + translate_stiffness_loss + laplacian_loss
                loss.backward()
                optimizer.step()

                deform_model.moveToLandMark()

                self.logger.addScalar('Loss/RotateStiffness', rotate_stiffness_loss.item(), step)
                self.logger.addScalar('Loss/TranslateStiffness', translate_stiffness_loss.item(), step)
                self.logger.addScalar('Loss/MatchingDist', masked_dist_loss.item(), step)
                self.logger.addScalar('Loss/Laplacian', laplacian_loss.item(), step)

                step += 1

            dist1, dist2 = self.chamfer_func(new_deformed_verts.unsqueeze(0), self.gt_points)[:2]
            l1_chamfer = toL1ChamferDistance(dist1, dist2)
            self.logger.addScalar('Metric/L1-Chamfer', l1_chamfer, step)

            if self.render:
                new_deformed_verts = deform_model.deform()
                new_deformed_mesh = toMesh(new_deformed_verts, template_triangles)
                new_deformed_mesh.compute_vertex_normals()
                image = renderGeometryImages(
                    new_deformed_mesh,
                    width=self.width,
                    height=self.height)
                self.logger.addImage('DeformedMesh', image, step)
                if self.save_result_folder_path is not None:
                    cv2.imwrite(save_deformed_image_folder_path + str(self.save_deform_image_idx) + '.jpg', image)
                    self.save_deform_image_idx += 1

            print(loss.item(), l1_chamfer)

            if i in self.milestones:
                w_idx += 1

        new_deformed_verts = deform_model.deform()
        self.updateTemplateVertices(new_deformed_verts)

        if self.render:
            save_video_file_path = self.save_result_folder_path + 'Deform.mp4'
            fps = 10
            toVideo(save_deformed_image_folder_path, save_video_file_path, fps)

            self.logger.addVideoFile('Deform', save_video_file_path, fps)
        return True

    def toDeformedTemplateMesh(self) -> o3d.geometry.TriangleMesh:
        deformed_template_mesh = self.template_mesh.toO3DMesh()

        transGeometry(
            deformed_template_mesh,
            self.gt_center,
            self.gt_scale,
            True)

        deformed_template_mesh.compute_vertex_normals()

        return deformed_template_mesh

    def saveDeformedTemplateMesh(self, save_mesh_file_path: str, overwrite: bool=False) -> bool:
        if os.path.exists(save_mesh_file_path):
            if not overwrite:
                return True

            removeFile(save_mesh_file_path)

        createFileFolder(save_mesh_file_path)

        deformed_template_mesh = self.toDeformedTemplateMesh()

        o3d.io.write_triangle_mesh(
            save_mesh_file_path,
            deformed_template_mesh,
            write_ascii=True,
        )
        return True
