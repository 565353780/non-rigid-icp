import torch
import numpy as np
import open3d as o3d
from tqdm import trange
from copy import deepcopy

from non_rigid_icp.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from non_rigid_icp.Method.icp import icp
from non_rigid_icp.Method.time import getCurrentTime
from non_rigid_icp.Method.render import render_geometry
from non_rigid_icp.Method.trans import toTensor, toNumpy
from non_rigid_icp.Method.utils import convert_mesh_to_pcl, laplacian_smoothing
from non_rigid_icp.Model.local_affine import AffineTransformLocal
from non_rigid_icp.Metric.chamfer import toL1ChamferDistance
from non_rigid_icp.Module.logger import Logger
from non_rigid_icp.Module.timer import Timer



def registration_mesh2pcl(
    template_mesh: o3d.geometry.TriangleMesh,
    target_pcl: o3d.geometry.PointCloud,
    inner_iter: int = 50,
    outer_iter: int = 100,
    milestones: list = [50, 80, 100, 110, 120, 130, 140],
    stiffness_weights: list = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2],
    laplacian_weight: float = 250,
    out_affine=False,
    device: str='cpu',
) -> bool:
    """
    Performs non-rigid iterative closest point (ICP) algorithm to align a source mesh to a target mesh.
    This function iteratively refines the alignment by minimizing the distance between the source and target meshes.

    Parameters:
    - template_mesh (o3d.geometry.TriangleMesh): The source mesh to be aligned to the target mesh.
    - target_mesh (o3d.geometry.TriangleMesh): The target mesh to which the source mesh is aligned.
    - config (dict): A dictionary containing configuration parameters for the non-rigid ICP algorithm, such as
      the number of iterations, regularization parameters, and convergence criteria.
    - out_affine (bool): A flag indicating whether to return the learned local affine transformation model.
    - in_affine (torch.nn.Module): A pre-trained local affine transformation model.
    - device (torch.device): The device on which to perform computations (e.g., 'cpu' or 'cuda').

    Returns:
    - o3d.geometry.TriangleMesh: The aligned source mesh after applying the non-rigid ICP algorithm.
    """
    log_folder_path = './logs/' + getCurrentTime() + '/'

    chamLoss = chamfer_3DDist()
    logger = Logger(log_folder_path)

    loop = trange(outer_iter)
    w_idx = 0

    phi = 30   # 水平旋转角度（绕 Z 轴）
    theta = 10  # 垂直仰角（相机高度）
    radius = 1.0  # 相机到物体的距离

    image = render_geometry(target_pcl, phi=phi, theta=theta, radius=radius)
    logger.addImage('GT', image, 0)

    transformation = icp(template_mesh, target_pcl)
    assert transformation is not None

    transformed_mesh = deepcopy(template_mesh).transform(transformation)

    image = render_geometry(transformed_mesh, phi=phi, theta=theta, radius=radius)
    logger.addImage('ICP', image, 0)

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

def registration_mesh2mesh(template_mesh: o3d.geometry.TriangleMesh,
                           target_mesh: o3d.geometry.TriangleMesh,
                           device: str='cpu'):
    target_pcl = convert_mesh_to_pcl(target_mesh)
    return registration_mesh2pcl(template_mesh,
                                 target_pcl,
                                 device=device)
