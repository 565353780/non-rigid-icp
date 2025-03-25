import torch


def get_adjacency_matrix(vertices: torch.Tensor, triangles: torch.Tensor):
    """
    input:
        mesh: open3d TriangleMesh object
    output:
        adjacency_matrix: sparse matrix of shape (num_verts, num_verts)
        degree_matrix: sparse matrix of shape (num_verts, num_verts)
    """
    # Create row and column indices for the adjacency matrix
    row_indices = triangles[:, [0, 1, 2, 0, 1]].flatten()
    col_indices = triangles[:, [1, 2, 0, 2, 0]].flatten()

    # Create the adjacency matrix as a sparse tensor
    adjacency_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=torch.ones(row_indices.shape[0]).to(vertices.device),
        size=(vertices.shape[0], vertices.shape[0]))

    # Compute degree values
    degree_values = torch.sparse.sum(adjacency_matrix, dim=1).to_dense()

    # Create the degree matrix as a sparse tensor
    idxs = torch.arange(degree_values.shape[0]).to(vertices.device)
    degree_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([idxs, idxs]),
        values=degree_values,
        size=(degree_values.shape[0], degree_values.shape[0]))

    return adjacency_matrix, degree_matrix

def toLaplacian(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    # Get adjacency matrix
    adjacency_matrix, degree_matrix = get_adjacency_matrix(vertices, triangles)

    # # Calculate degree matrix
    # degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))

    # Calculate Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Calculate Laplacian of each vertex
    laplacian = torch.matmul(laplacian_matrix, vertices)

    return laplacian

def laplacian_smoothing(vertices: torch.Tensor, triangles: torch.Tensor):
    laplacian = toLaplacian(vertices, triangles)

    # Calculate Laplacian loss
    laplacian_loss = torch.sum(laplacian**2)

    # new_vertices = vertices - 0.5 * laplacian

    return laplacian_loss


def toLaplacianLoss(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    source_laplacian: torch.Tensor,
) -> torch.Tensor:
    laplacian = toLaplacian(vertices, triangles)

    laplacian_dists2 = (laplacian - source_laplacian) ** 2

    laplacian_loss = torch.mean(laplacian_dists2)

    return laplacian_loss
