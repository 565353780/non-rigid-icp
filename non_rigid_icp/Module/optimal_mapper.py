import os
import torch
import numpy as np
import open3d as o3d
from tqdm import trange
from typing import Union
from copy import deepcopy

from non_rigid_icp.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from non_rigid_icp.Method.icp import icp
from non_rigid_icp.Method.pcd import toPointCloud
from non_rigid_icp.Method.time import getCurrentTime
from non_rigid_icp.Method.render import renderGeometry
from non_rigid_icp.Method.trans import toNormalizeTransform, toTensor, toNumpy, transGeometry, transMesh
from non_rigid_icp.Method.utils import convert_mesh_to_pcl, laplacian_smoothing
from non_rigid_icp.Model.local_affine import AffineTransformLocal
from non_rigid_icp.Metric.chamfer import toL1ChamferDistance
from non_rigid_icp.Module.logger import Logger
from non_rigid_icp.Module.timer import Timer
from non_rigid_icp.Method.render import renderColoredMeshes


class OptimalMapper(object):
    def __init__(
        self,
        inner_iter: int = 50,
        outer_iter: int = 100,
        milestones: list = [50, 80, 100, 110, 120, 130, 140],
        stiffness_weights: list = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2],
        laplacian_weight: float = 250,
        device: str='cpu',
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.inner_iter = inner_iter
        self.outer_iter = outer_iter
        self.milestones = milestones

        self.stiffness_weights = stiffness_weights
        self.laplacian_weight = laplacian_weight

        self.device = device

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.logger = Logger()

        self.chamfer_func = chamfer_3DDist()

        self.initRecords()

        self.phi = 30   # 水平旋转角度（绕 Z 轴）
        self.theta = 10  # 垂直仰角（相机高度）
        self.radius = 1.0  # 相机到物体的距离

        self.gt_points = None
        self.gt_pcd = None
        self.gt_center = None
        self.gt_scale = None
        return

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

    def loadGTPoints(self, gt_points: Union[torch.Tensor, np.ndarray]) -> bool:
        if isinstance(gt_points, np.ndarray):
            self.gt_points = gt_points
        else:
            self.gt_points = toNumpy(gt_points)

        self.gt_points = self.gt_points.reshape(-1, 3)

        self.gt_pcd = toPointCloud(self.gt_points[0])

        self.gt_center, self.gt_scale = toNormalizeTransform(toNumpy(self.gt_points[0]))

        transGeometry(self.gt_pcd, self.gt_center, self.gt_scale)

        image = renderGeometry(
            self.gt_pcd,
            phi=self.phi,
            theta=self.theta,
            radius=self.radius)
        self.logger.addImage('GT', image)
        return True

    def loadGTPcd(self, gt_pcd: o3d.geometry.PointCloud) -> bool:
        return self.loadGTPoints(gt_pcd.points)

    def loadGTMesh(self, gt_mesh: o3d.geometry.TriangleMesh) -> bool:
        return self.loadGTPoints(gt_mesh.vertices)

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

        gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_file_path)
        return self.loadGTMesh(gt_mesh)

    def estimateInitPose(
        self,
        source_mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:
        transformation = icp(source_mesh, self.gt_pcd)
        assert transformation is not None

        transformed_mesh = deepcopy(template_mesh).transform(transformation)

        image = render_geometry(transformed_mesh, phi=phi, theta=theta, radius=radius)
        logger.addImage('ICP', image, 0)


        return True

    def registration_mesh2pcl(
        self,
        template_mesh: o3d.geometry.TriangleMesh,
        target_mesh: o3d.geometry.TriangleMesh,
    ) -> bool:
        loop = trange(self.outer_iter)
        w_idx = 0

        transformed_vertex = toTensor(transformed_mesh.vertices, device)

        template_vertex = toTensor(template_mesh.vertices, device)

        target_vertex_np = np.asarray(target_pcl.points)
        target_vertex = toTensor(target_vertex_np, device).unsqueeze(0)

        # define the transformation model (Local Affine Transform)
        triangles = np.asarray(template_mesh.triangles)

        edges = np.vstack(
            [triangles[:, :2], triangles[:, 1:3], triangles[:, [0, 2]]])
        edges = np.sort(edges, axis=1)  # sort the vertices for each edge
        edges = np.unique(edges, axis=0)  # remove duplicate edges

        template_edges = toTensor(edges, device, torch.long)

        local_affine_model = AffineTransformLocal(
            template_vertex.shape[0], template_edges).to(device)

        # define optimizer
        optimizer = torch.optim.AdamW([{
            'params': local_affine_model.parameters()
        }],
                                    lr=1e-4,
                                    amsgrad=True)

        step = 0
        for i in loop:
            # just uses linear transformation based on learned parameters and also uses stiffness term
            new_deformed_verts, stiffness = local_affine_model(
                transformed_vertex, return_stiff=True)

            dist1, dist2, idx1, idx2 = chamLoss(new_deformed_verts.unsqueeze(0), target_vertex)

            if i == 0:
                inner_loop = range(4)
            else:
                inner_loop = range(inner_iter)

            close_points = target_vertex[0, idx1.squeeze(0)]

            # enter inner loop
            for _ in inner_loop:
                optimizer.zero_grad()

                vert_distance = (new_deformed_verts - close_points)**2

                # we need a sum over vector components for L2 norm. Set 0.04 as threshold
                vert_distance_mask = torch.sum(vert_distance, dim=1) < 0.04**2

                weight_mask = vert_distance_mask.unsqueeze(-1)

                vert_distance = weight_mask * vert_distance

                # This is the first term of the Loss function
                vert_sum = torch.sum(vert_distance)

                # Stiffness loss  term. L2 loss for stiffness weighted by stiffness weights
                stiffness_sum = torch.sum(stiffness) * stiffness_weights[w_idx]

                # Laplacian smoothing loss term
                # It describes how a vertex deviates from the average of its neighbors
                template_mesh.vertices = o3d.utility.Vector3dVector(
                    toNumpy(new_deformed_verts, np.float64))

                laplacian_loss = laplacian_smoothing(template_mesh)

                # Laplacian weight
                laplacian_loss = laplacian_loss * laplacian_weight

                # sum up all the loss terms
                loss = torch.sqrt(vert_sum + stiffness_sum) + laplacian_loss
                loss.backward()
                optimizer.step()
                # here we again use transformed_vertex (obtained as a result of initial rigid transformation
                # of landmarks).

                new_deformed_verts, stiffness = local_affine_model(
                    transformed_vertex, return_stiff=True)

                template_mesh.vertices = o3d.utility.Vector3dVector(
                    toNumpy(new_deformed_verts, np.float64))
                new_deformed_mesh = template_mesh

                logger.addScalar('Loss/Stiffness', stiffness_sum.item(), step)
                logger.addScalar('Loss/MatchingDist', vert_sum.item(), step)
                logger.addScalar('Loss/Laplacian', laplacian_loss.item(), step)

                step += 1

            l1_chamfer = toL1ChamferDistance(dist1, dist2)
            logger.addScalar('Metric/L1-Chamfer', l1_chamfer, step)

            image = render_geometry(template_mesh, phi=phi, theta=theta, radius=radius)
            logger.addImage('DeformedMesh', image, step)

            # final loss calculation in outer loop
            print(l1_chamfer, stiffness_sum.item(), vert_sum.item(), laplacian_loss.item())

            if i in milestones:
                w_idx += 1

        template_mesh.vertices = o3d.utility.Vector3dVector(
            toNumpy(new_deformed_verts, np.float64))
        new_deformed_mesh = template_mesh
        if out_affine:
            return new_deformed_mesh, local_affine_model
        else:
            return new_deformed_mesh

    def registration_mesh2mesh(self, template_mesh: o3d.geometry.TriangleMesh,
                            target_mesh: o3d.geometry.TriangleMesh,
                            device: str='cpu'):
        target_pcl = convert_mesh_to_pcl(target_mesh)
        return self.registration_mesh2pcl(template_mesh,
                                    target_pcl,
                                    device=device)
