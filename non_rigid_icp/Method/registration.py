import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from copy import deepcopy
from scipy.spatial import KDTree

from non_rigid_icp.Method.icp import icp
from non_rigid_icp.Method.trans import toTensor, toNumpy
from non_rigid_icp.Method.utils import convert_mesh_to_pcl, laplacian_smoothing
from non_rigid_icp.Model.local_affine import AffineTransformLocal


def registration_mesh2pcl(
    template_mesh: o3d.geometry.TriangleMesh,
    target_pcl: o3d.geometry.PointCloud,
    inner_iter: int = 50,
    outer_iter: int = 100,
    milestones: list = [50, 80, 100, 110, 120, 130, 140],
    stiffness_weights: list = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2],
    laplacian_weight: float = 250,
    out_affine=False,
    in_affine=None,
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
    loop = tqdm(range(outer_iter))
    w_idx = 0

    transformation = icp(template_mesh, target_pcl)
    assert transformation is not None

    transformed_mesh = deepcopy(template_mesh).transform(transformation)
    transformed_vertex = toTensor(transformed_mesh.vertices, device)

    template_vertex = toTensor(template_mesh.vertices, device)

    target_vertex_np = np.asarray(target_pcl.points)
    target_vertex = toTensor(target_vertex_np, device)

    # build a KDTree for efficient nearest neighbor search
    tree = KDTree(target_vertex_np)

    # define the transformation model (Local Affine Transform)
    triangles = np.asarray(template_mesh.triangles)

    edges = np.vstack(
        [triangles[:, :2], triangles[:, 1:3], triangles[:, [0, 2]]])
    edges = np.sort(edges, axis=1)  # sort the vertices for each edge
    edges = np.unique(edges, axis=0)  # remove duplicate edges

    template_edges = toTensor(edges, device, torch.long)

    if in_affine is None:
        local_affine_model = AffineTransformLocal(template_vertex.shape[0],
                                                  template_edges).to(device)
    else:
        local_affine_model = in_affine

    # define optimizer
    optimizer = torch.optim.AdamW([{
        'params': local_affine_model.parameters()
    }],
                                  lr=1e-4,
                                  amsgrad=True)

    for i in loop:
        # just uses linear transformation based on learned parameters and also uses stiffness term
        new_deformed_verts, stiffness = local_affine_model(
            transformed_vertex, return_stiff=True)

        old_verts = new_deformed_verts
        new_deformed_mesh = template_mesh

        # set new template vertices based on transformation
        new_deformed_mesh.vertices = o3d.utility.Vector3dVector(
            toNumpy(new_deformed_verts, np.float64))

        new_deformed_verts_np = toNumpy(new_deformed_verts)

        # Query the KDTree for the nearest neighbor to find closeset points on target mesh/point cloud
        distances, indices = tree.query(new_deformed_verts_np, k=1)

        indices = toTensor(indices, device, torch.int64)
        close_points = target_vertex[indices, :]

        if (i == 0) and (in_affine is None):
            inner_loop = range(4)
        else:
            inner_loop = range(inner_iter)

        # enter inner loop
        for _ in inner_loop:
            optimizer.zero_grad()

            vert_distance = (new_deformed_verts - close_points)**2

            # we need a sum over vector components for L2 norm. Set 0.04 as threshold
            vert_distance_mask = torch.sum(vert_distance, dim=1) < 0.04**2

            weight_mask = vert_distance_mask.unsqueeze(-1)

            # multipley mask by vert_distance to select vertex that match conditions,
            # specifically that the distance should be less than 0.04**2
            # multiplying False/True by number gives 0/1
            vert_distance = weight_mask * vert_distance

            # This is the first term of the Loss function
            vert_sum = torch.sum(vert_distance)

            # Stiffness loss  term. L2 loss for stiffness weighted by stiffness weights
            stiffness_sum = torch.sum(stiffness) * stiffness_weights[w_idx]

            # Laplacian smoothing loss term
            # It describes how a vertex deviates from the average of its neighbors
            laplacian_loss = laplacian_smoothing(new_deformed_mesh)

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

        # final loss calculation in outer loop
        distance = torch.mean(
            torch.sqrt(torch.sum((old_verts - new_deformed_verts)**2, dim=1)))
        print(distance.item(), stiffness_sum.item(), vert_sum.item(), laplacian_loss.item())

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
