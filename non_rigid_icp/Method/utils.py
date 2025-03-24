import torch
import numpy as np
import open3d as o3d


def convert_mesh_to_pcl(in_mesh: o3d.geometry.TriangleMesh):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(np.asarray(in_mesh.vertices))
    return pcl

def get_adjacency_matrix(mesh: o3d.geometry.TriangleMesh):
    """
    input:
        mesh: open3d TriangleMesh object
    output:
        adjacency_matrix: sparse matrix of shape (num_verts, num_verts)
        degree_matrix: sparse matrix of shape (num_verts, num_verts)
    """
    # Get the triangles of the mesh
    triangles = np.array(mesh.triangles)  # Convert to numpy array first
    triangles = torch.from_numpy(triangles)  # Then convert to tensor

    # Create row and column indices for the adjacency matrix
    row_indices = triangles[:, [0, 1, 2, 0, 1]].flatten()
    col_indices = triangles[:, [1, 2, 0, 2, 0]].flatten()

    # Create the adjacency matrix as a sparse tensor
    adjacency_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=torch.ones(len(row_indices)),
        size=(len(mesh.vertices), len(mesh.vertices)))

    # Compute degree values
    degree_values = torch.sparse.sum(adjacency_matrix, dim=1).to_dense()

    # Create the degree matrix as a sparse tensor
    degree_matrix = torch.sparse_coo_tensor(indices=torch.stack([
        torch.arange(degree_values.shape[0]),
        torch.arange(degree_values.shape[0])
    ]),
                                            values=degree_values,
                                            size=(degree_values.shape[0],
                                                  degree_values.shape[0]))

    return adjacency_matrix, degree_matrix

def laplacian_smoothing(mesh: o3d.geometry.TriangleMesh, lamb=0.5):
    # Get adjacency matrix
    adjacency_matrix, degree_matrix = get_adjacency_matrix(mesh)

    # Convert mesh vertices to PyTorch tensor
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)

    # # Calculate degree matrix
    # degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))

    # Calculate Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Calculate Laplacian of each vertex
    laplacian = torch.matmul(laplacian_matrix, vertices)

    # Calculate Laplacian loss
    laplacian_loss = torch.sum(laplacian**2)

    # Move each vertex towards the average position of its neighbors
    new_vertices = vertices - lamb * laplacian

    # Update mesh vertices
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices.detach().numpy())

    return laplacian_loss
