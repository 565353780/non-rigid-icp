import torch
import numpy as np
import open3d as o3d


def mesh_boundary(in_faces: torch.LongTensor, num_verts: int):
    '''
    input:
        in edges: N * 3, is the vertex index of each face, where N is number of faces
        num_verts: the number of vertexs mesh
    return:
        boundary_mask: bool tensor of num_verts, if true, point is on the boundary, else not
    '''
    in_x = in_faces[:, 0]
    in_y = in_faces[:, 1]
    in_z = in_faces[:, 2]
    in_xy = in_x * (num_verts) + in_y
    in_yx = in_y * (num_verts) + in_x
    in_xz = in_x * (num_verts) + in_z
    in_zx = in_z * (num_verts) + in_x
    in_yz = in_y * (num_verts) + in_z
    in_zy = in_z * (num_verts) + in_y
    in_xy_hash = torch.minimum(in_xy, in_yx)
    in_xz_hash = torch.minimum(in_xz, in_zx)
    in_yz_hash = torch.minimum(in_yz, in_zy)
    in_hash = torch.cat((in_xy_hash, in_xz_hash, in_yz_hash), dim=0)
    output, count = torch.unique(in_hash, return_counts=True, dim=0)
    boundary_edge = output[count == 1]
    boundary_vert1 = boundary_edge // num_verts
    boundary_vert2 = boundary_edge % num_verts
    boundary_mask = torch.zeros(num_verts).bool()
    boundary_mask[boundary_vert1] = True
    boundary_mask[boundary_vert2] = True
    return boundary_mask

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
